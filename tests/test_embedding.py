import unittest
from unittest import mock

from recommendation_training.embedding import load_encoder


class FrozenEncoderTests(unittest.TestCase):
    def test_load_encoder_explicitly_disables_parameter_gradients(self):
        processor = object()
        model = mock.Mock()
        model.to.return_value = model
        model.requires_grad_.return_value = model
        model.eval.return_value = model

        with mock.patch(
            "recommendation_training.embedding.AutoProcessor.from_pretrained",
            return_value=processor,
        ), mock.patch(
            "recommendation_training.embedding.AutoModel.from_pretrained",
            return_value=model,
        ):
            loaded_processor, loaded_model, device = load_encoder(device="cpu")

        self.assertIs(loaded_processor, processor)
        self.assertIs(loaded_model, model)
        self.assertEqual(device, "cpu")
        model.to.assert_called_once_with("cpu")
        model.requires_grad_.assert_called_once_with(False)
        model.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
