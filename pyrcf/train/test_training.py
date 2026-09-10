"""Regression checks for upstream RCF semantics and local integration."""

import copy
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ..models import RCF
from .checkpoint import restore_training_checkpoint, save_training_checkpoint
from .data import BGR_MEAN, EdgeManifestDataset
from .loss import RCFLoss
from .optimization import create_rcf_optimizer
from .runner import split_indices
from .trainer import RCFTrainer


class TinySixOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, image):
        return [torch.sigmoid(image * self.weight + i * 0.1) for i in range(6)]


class TrainingTests(unittest.TestCase):
    def test_black_input_pixels_are_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.full((9, 9, 3), 100, dtype=np.uint8)
            image[0, :2] = 0
            image[0, 2] = [0, 0, 1]
            label = np.zeros((9, 9), dtype=np.uint8)
            label[0, 1:4] = 255
            for name, array in (("image.png", image), ("label.png", label)):
                self.assertTrue(cv2.imwrite(str(root/name), array))
            manifest = root/"train.lst"
            manifest.write_text("image.png\tlabel.png\n")
            _, target = EdgeManifestDataset(manifest, crop_size=None)[0]
            self.assertEqual(target[0, 0, :4].tolist(), [2, 2, 1, 1])
            outputs = [torch.full_like(target, 0.5, requires_grad=True) for _ in range(6)]
            loss, parts = RCFLoss()(outputs, target)
            loss.backward()
            self.assertEqual(parts["positive_pixels"].item(), 2)
            for output in outputs:
                self.assertTrue((output.grad[0, 0, :2] == 0).all())
                self.assertNotEqual(output.grad[0, 0, 2].item(), 0)

    def test_ignored_group_does_not_update_weights(self):
        model = TinySixOutput()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.1)
        trainer = RCFTrainer(model, optimizer, RCFLoss(), "cpu", accumulation_steps=2)
        image = torch.ones(1, 1, 1, 2)
        trainer.train_epoch([(image, torch.tensor([[[[0., 1.]]]]))], 1)
        before = model.weight.detach().clone()
        momentum = optimizer.state[model.weight]["momentum_buffer"].clone()
        trainer.train_epoch([(image, torch.full_like(image, 2))] * 3, 2)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[model.weight]["momentum_buffer"], momentum)

    def test_upstream_bce_and_ignored_gradients(self):
        labels = torch.tensor([[[[0., 1., 2., 0.]]]])
        prediction = torch.tensor([[[[0.2, 0.7, 0.8, 0.4]]]], requires_grad=True)
        loss, _ = RCFLoss()([prediction] * 6, labels)
        expected = -6 * ((1.1/3) * (torch.log(1-prediction[0,0,0,0])
                    + torch.log(1-prediction[0,0,0,3]))
                    + (2/3) * torch.log(prediction[0,0,0,1]))
        torch.testing.assert_close(loss, expected * (320 * 320 / 3))
        loss.backward()
        self.assertEqual(prediction.grad[0,0,0,2].item(), 0)
        for value in (0., 1., 2.):
            out = torch.full_like(labels, 0.5, requires_grad=True)
            zero, _ = RCFLoss()([out]*6, torch.full_like(labels, value))
            zero.backward()
            self.assertEqual(zero.item(), 0)
            self.assertEqual(out.grad.abs().sum().item(), 0)

    def test_loss_scale_is_independent_of_crop_area(self):
        labels = torch.tensor([[[[0., 1.], [1., 0.]]]])
        prediction = torch.tensor([[[[.2, .7], [.8, .3]]]])
        small, _ = RCFLoss()([prediction] * 6, labels)
        large_labels = labels.repeat(1, 1, 8, 8)
        large_prediction = prediction.repeat(1, 1, 8, 8)
        large, _ = RCFLoss()([large_prediction] * 6, large_labels)
        torch.testing.assert_close(large, small)

    def test_official_optimizer_groups(self):
        model = RCF()
        optimizer = create_rcf_optimizer(model)
        parameters = {id(p): name for name, p in model.named_parameters()}
        seen = []
        scales = {"conv1-4": 1, "conv5": 100, "down": .1, "side": .01, "fuse": .001}
        for group in optimizer.param_groups:
            family, kind = group["name"].split(".")
            self.assertAlmostEqual(group["lr"], 1e-6 * scales[family] * (2 if kind == "bias" else 1))
            self.assertEqual(group["weight_decay"], 0 if kind == "bias" else 2e-4)
            seen.extend(id(p) for p in group["params"])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(parameters))
        self.assertEqual(len(optimizer.param_groups), 10)

    def test_accumulation_and_resume(self):
        model = TinySixOutput()
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01, momentum=.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.1)
        label = torch.tensor([[[[0., 1.]]]])
        batches = [(torch.tensor([[[[float(i), float(i+1)]]]]), label) for i in range(1, 4)]
        trainer = RCFTrainer(model, optimizer, RCFLoss(), "cpu", accumulation_steps=2)
        trainer.train_epoch(batches, 1)
        # Explicit upstream-style accumulation for the full group and final tail.
        for group in (batches[:2], batches[2:]):
            ref_optimizer.zero_grad()
            for image, target in group:
                loss, _ = RCFLoss()(reference(image), target)
                (loss / len(group)).backward()
            ref_optimizer.step()
        torch.testing.assert_close(model.weight, reference.weight)
        scheduler.step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"resume.pth"
            save_training_checkpoint(path, 1, model, optimizer, scheduler)
            restored = TinySixOutput()
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=.01, momentum=.9)
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1, gamma=.1)
            epoch = restore_training_checkpoint(path, restored, restored_optimizer, restored_scheduler)
            self.assertEqual(epoch, 1)
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
            resumed = RCFTrainer(restored, restored_optimizer, RCFLoss(), "cpu", accumulation_steps=2)
            trainer.train_epoch(batches, 2)
            resumed.train_epoch(batches, 2)
            torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)
            before = restored.weight.detach().clone()
            resumed.validate(batches, 2)
            torch.testing.assert_close(before, restored.weight)

    def test_official_preprocessing_masks_and_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.full((9, 10, 3), (30, 80, 140), dtype=np.uint8)
            label = np.zeros((9, 10), dtype=np.uint8)
            label[0, :4] = [0, 127, 128, 255]
            mask = np.full_like(label, 255)
            mask[0, 3] = 0
            for name, array in (("image.png", image), ("label.png", label), ("mask.png", mask)):
                self.assertTrue(cv2.imwrite(str(root/name), array))
            manifest = root/"train.lst"
            manifest.write_text("image.png\tlabel.png\tmask.png\n")
            dataset = EdgeManifestDataset(manifest, crop_size=None)
            tensor, target = dataset[0]
            np.testing.assert_allclose(tensor[:,0,0].numpy(), image[0,0].astype(np.float32)-BGR_MEAN)
            self.assertEqual(target[0,0,:4].tolist(), [0, 2, 1, 2])
            tensor, target = EdgeManifestDataset(manifest, crop_size=16, crop_mode="center")[0]
            self.assertEqual(tensor.shape, (3,16,16))
            self.assertTrue((target[:,9:,:] == 2).all())
            self.assertTrue((target[:,:,10:] == 2).all())
        self.assertEqual(split_indices(1, .9, 42), ([0], []))
        train, val = split_indices(10, .1, 42)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(train)+len(val), 10)


if __name__ == "__main__":
    unittest.main()
