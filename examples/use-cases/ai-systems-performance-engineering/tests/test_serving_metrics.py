"""Synthetic accounting checks; these fixtures are not hardware measurements."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
from serving_metrics import METRICS, compute_mbu, decode_record, model_bytes


class ServingMetricTests(unittest.TestCase):
    def test_decode_counts_exclude_prefill_and_reject_interleaved_work(self):
        before = dict.fromkeys(METRICS,0)
        after = dict(zip(METRICS,[8,2,6,.3]))
        row = decode_record(before,after,[{'output_tokens':4},{'output_tokens':4}])
        self.assertEqual(row['decode_steps'],6)
        self.assertAlmostEqual(row['server_mean_itl_seconds'],.05)
        after[METRICS[0]]=9
        with self.assertRaises(ValueError): decode_record(before,after,[{'output_tokens':4},{'output_tokens':4}])

    def test_weight_headers_include_replication_and_reject_wrong_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'config.json').write_text(json.dumps(dict(model_type='qwen2',tie_word_embeddings=True,num_key_value_heads=2,num_attention_heads=16)))
            header={'model.embed_tokens.weight':dict(dtype='BF16',shape=[4,2],data_offsets=[0,16]),'model.norm.weight':dict(dtype='BF16',shape=[2],data_offsets=[16,20])}
            raw=json.dumps(header).encode(); (p/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*20)
            result=model_bytes(p,2)
            self.assertEqual(result['checkpoint_weight_bytes'],20)
            self.assertEqual(result['weights_bytes_per_decode_step'],24)
            header['model.norm.weight']['dtype']='F32'
            raw=json.dumps(header).encode(); (p/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw)
            with self.assertRaises(ValueError): model_bytes(p,2)

    def test_mbu_uses_each_replicas_decode_duration_and_rejects_gpu_duplicates(self):
        serving=dict(concurrency=1,instance_type='synthetic',decode_measurements={'a':dict(decode_steps=6,server_decode_seconds=.3)})
        weights=dict(tensor_parallel_size=2,weights_bytes_per_decode_step=20)
        sample=dict(correctness='passed',method='streaming_read_only_cg',instance_type='synthetic',median_bytes_per_second=1000,gpu_uuid_hex='a')
        bandwidth={'a':[sample,sample|{'gpu_uuid_hex':'b'}]}
        result=compute_mbu(serving,weights,bandwidth)
        self.assertAlmostEqual(result['mbu_weights_only_ratio'],.2)
        with self.assertRaises(ValueError): compute_mbu(serving,weights,{'a':[sample,sample]})
        with self.assertRaises(ValueError): compute_mbu(serving|{'concurrency':4},weights,bandwidth)


if __name__=='__main__': unittest.main()
