# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_extra.json
- Mode: ollama
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.49 | 0.62 | 0.13 |
| Avg attempts | 1 | 1.92 | 0.92 |
| Avg token estimate | 827.38 | 2445.13 | 1617.75 |
| Avg latency (ms) | 4927.13 | 17534.74 | 12607.61 |

## Failure modes
```json
{
  "react": {
    "none": 49,
    "entity_drift": 47,
    "wrong_final_answer": 4
  },
  "all": {
    "none": 111,
    "entity_drift": 53,
    "wrong_final_answer": 6,
    "looping": 30
  },
  "reflexion": {
    "none": 62,
    "looping": 30,
    "wrong_final_answer": 2,
    "entity_drift": 6
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json

## Discussion
Reflexion đạt EM 0.62 so với ReAct 0.49 (delta +0.13) trên 100 câu HotpotQA. Cái giá phải trả: trung bình 1.92 attempts/câu (ReAct luôn là 1), tokens tăng 2.96× và latency tăng 3.56×. Phân tích failure_modes cho thấy ReAct phân bổ lỗi như sau: {'none': 49, 'entity_drift': 47, 'wrong_final_answer': 4}, trong khi Reflexion còn lại: {'none': 62, 'looping': 30, 'wrong_final_answer': 2, 'entity_drift': 6}. Sau reflection, lỗi phổ biến nhất chưa xử lý được là 'looping' (30 câu) — cho thấy evaluator đôi khi không chỉ ra đúng thiếu sót, hoặc reflector đưa ra chiến thuật quá chung chung nên actor vẫn lặp lại lỗi. Về mặt vận hành, với budget đủ dư, đánh đổi này hợp lý cho câu hỏi multi-hop. Với budget khắt khe, cần adaptive_max_attempts hoặc memory_compression để giảm chi phí. Model sử dụng là qwen2.5:3b-instruct qua Ollama; chất lượng evaluator/reflector giới hạn bởi kích thước model.
