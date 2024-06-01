import unittest

import torch
import transformers

from modom.mixture_of_depth import MixtureOfDepth, convert_hf_model

input_string = 'Given the following HTML provide the bounding box\\n <button backend_node_id="661"></button>'


class TestMixtureOfDepth(unittest.TestCase):
    def setUp(self):
        self.model = transformers.AutoModel.from_pretrained("gpt2")
        self.processor = transformers.AutoTokenizer.from_pretrained("gpt2")
        self.mod = MixtureOfDepth(self.model.transformer.h[0], 0.125)

    def test_forward(self):
        x = torch.rand(1, 10, self.model.config.n_embd)
        attention_mask = torch.ones(1, 10)
        position_ids = torch.arange(10).unsqueeze(0)
        past_key_value = (
            torch.rand(1, 2, 10, self.model.config.n_embd),
            torch.rand(1, 2, 10, self.model.config.n_embd),
        )
        output_attentions = False
        use_cache = False
        cache_position = None

        output = self.mod(x, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)
        self.assertIsInstance(output, tuple)
        self.assertEqual(output[0].shape, x.shape)


class TestConvertHF(unittest.TestCase):
    def setUp(self):
        self.device_map = "auto"

    def test_apply_mod_to_llama(self):
        """
        test llama3 with multiple inputs
        """
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device_map)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        inputs = tokenizer(text=[input_string, input_string], return_tensors="pt")
        model = convert_hf_model(model)
        inputs.to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(hasattr(outputs, "logits"))

    def test_apply_mod_to_persimmon(self):
        """
        test fuyu/persimmon which have outdated llama arch/methods
        """
        # for testing locally drop model size by a lot
        model_id = "adept/persimmon-8b-base"

        model_config = transformers.AutoConfig.from_pretrained(
            model_id, hidden_size=128, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4
        )

        model = transformers.AutoModelForCausalLM.from_config(model_config)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        inputs = tokenizer(text=[input_string, input_string], return_tensors="pt")
        inputs.to(model.device)

        model = convert_hf_model(model, skip_position_ids=True)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(hasattr(outputs, "logits"))
        self.assertEqual(model.__class__.__name__, "PersimmonMoDForCausalLM")

    def test_apply_mod_to_gemma(self):
        """
        test paligemma since it is a ConditionalGeneration model as opposed to CausalLM
        """
        model_id = "google/paligemma-3b-ft-docvqa-896"
        model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map=self.device_map)
        processor = transformers.PaliGemmaProcessor.from_pretrained(model_id)

        model = convert_hf_model(model)

        inputs = processor(text=input_string, images=torch.rand(1, 3, 1280, 1080))
        inputs.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(hasattr(outputs, "logits"))


if __name__ == "__main__":
    unittest.main()
