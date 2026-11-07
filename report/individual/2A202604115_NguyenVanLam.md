# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604115
- **Họ và tên:** Nguyễn Văn Lâm
- **SEED:** 5103

## 1. Bằng chứng day01_lab

| Tệp | Có? |
|---|---|
| `day01_lab.ipynb` | ☑ |
| `lab1_report.json` | ☑ |
| `hand_calc.jpg` | ☑ |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 của mình. Các hàng LLM chỉ đo trên 40 tin SUB nên không có F1 test.

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD / 1k tin |
|---|---:|---:|---:|---:|
| 1. BoW + LogReg (tự code) | 0.903 | 0.824 | 0.090 | 0 |
| 2. BoW + MLP (deep learning) | 0.935 | 0.923 | 0.344 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.904 | 0.889 | 57.956 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.947 | 0.919 | 31.258 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.963 | 0.919 | 32.512 | 0 |
| 4a. LLM nhỏ zero-shot (Haiku) | – | 0.923 | 1000.069 | 0.0101 |
| 4b. LLM lớn zero-shot (Opus) | – | 0.974 | 1590.094 | 0.4025 |
| 4c. LLM nhỏ few-shot (Haiku) | – | 0.895 | 1073.156 | 0.0229 |

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất so với kết quả:

**1. Few-shot (`llm_fewshot`)**
- **Mình đoán:** thêm 4 ví dụ từ tập train vào prompt thì F1 tăng nhẹ, chi phí tăng theo số input token.
- **Ra gì:** F1 SUB **giảm** từ 0.923 xuống **0.895**, chi phí tăng từ $0.0101 lên $0.0229 mỗi 1k tin (gấp ~2.3 lần).
- **Vì sao lệch:** Chi phí đoán đúng, F1 đoán sai hướng. Recall của Haiku giảm từ 0.90 xuống 0.85, tức chỉ mất thêm khoảng 1 tin spam trên SUB (suy ra từ recall, SUB có 20 spam), nên phần lớn là nhiễu của tập 40 tin. Giả thuyết của mình: 4 ví dụ ngẫu nhiên chưa đại diện, có thể kéo model về phía "ham" ở các tin kiểu khuyến mãi.

**2. Bỏ ReLU (`dl_relu`)**
- **Mình đoán:** hai lớp Linear gộp thành một phép tuyến tính, tương đương logistic regression, F1 giảm về gần 0.903.
- **Ra gì:** F1 vẫn **0.935**, bằng đúng khi có ReLU.
- **Vì sao lệch:** Mình nhầm "tương đương về dạng hàm" với "cho kết quả giống LogReg đã train". Mạng không ReLU vẫn có 128,129 tham số và được huấn luyện bằng PyTorch với optimizer, batch và số epoch khác với LogReg tự code, nên điểm hội tụ khác nhau.

**3. Tin tricky: Encoder và BoW (`emb_tricky`)**
- **Mình đoán:** Encoder + LogReg đúng nhiều hơn BoW vì BoW bị đánh lừa bởi "free", "won", "prize" trong tin ham và bỏ lọt spam né từ khóa.
- **Ra gì:** cả hai đều đúng **5/6**, cùng sai tin phishing "Last chance! Your account rewards expire tonight...".
- **Vì sao lệch:** BoW không bị đánh lừa như mình nghĩ vì các từ đời thường trong cùng tin ("hey", "lunch", "mum", "cake"...) có trọng số âm đủ lớn để kéo điểm về ham. Tin phishing không có từ khóa spam rõ và cũng không có chủ đề "trúng thưởng" nên Encoder cũng bỏ lọt. Chỉ có 6 tin nên không kết luận được model nào hơn. Trên toàn test, MiniLM + LogReg (0.947) vẫn cao hơn BoW + LogReg (0.903).

## 4. Phân tích 50 case khó

Bộ HARD_EVAL có 50 case viết tay, mình thêm 5 case nên chạy trên 55 case.

**Model tụt nhiều nhất: BoW + MLP**, F1 từ 0.923 (SUB) xuống **0.439** trên bộ khó (giảm 0.48; BoW + LogReg giảm từ 0.824 xuống 0.359, tức giảm 0.47). Tụt mạnh nhất ở hai nhóm:

| Nhóm | BoW + MLP | BoW + LogReg | Haiku (4a) |
|---|---:|---:|---:|
| Né từ khóa (6 case) | 0% | 17% | 100% |
| Spam giọng tự nhiên (10 case) | 10% | 10% | 100% |

Hai case cụ thể (cả 6 case né từ khóa đều bị BoW + MLP đoán sai):

- **Case #29, nhóm né từ khóa** — "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m", nhãn thật là spam.
  **Cơ chế:** BoW chỉ tra từ trong từ điển 2000 từ. Các từ đã bị thay số như "c0ngr4tulat10ns", "w0n", "cl4im", "g1ft" đều không nằm trong từ điển nên bị bỏ qua. Phần còn lại ("a", "card", "£1000") không đẩy về spam nên mô hình đoán ham. Mô hình không nhận ra "w0n" là "won".
- **Case #34, nhóm né từ khóa** — "W.I.N a brand new iPh0ne 15 - just rep|y Y to this message", nhãn thật là spam.
  **Cơ chế:** chèn dấu chấm và đổi ký tự ("W.I.N", "rep|y", "iPh0ne") làm từ khóa spam bị tách thành các token rời hoặc ký tự lạ nên không khớp từ điển. Các từ còn lại ("brand", "new", "just", "this", "message") đều trung tính, mô hình không thấy bằng chứng nào cho spam. Đây là cơ chế mình suy ra từ cách BoW hoạt động, chưa xem điểm từng từ của case này.

Nhóm spam giọng tự nhiên (ví dụ #11 "Hey it's Jess from the gym... just reply JOIN") sai vì cùng lý do: câu chữ thân mật không chứa từ khóa spam, BoW cộng điểm từng từ và không hiểu ý đồ khuyến mãi.

Embedding khá hơn ở né từ khóa (83%) nhưng yếu ở nhóm "nói về spam" (25–50%). Haiku đúng 96% tổng, prompt phòng thủ (4d) không cải thiện injection (89%) và làm tụt nhóm "tưởng spam nhưng không phải" từ 100% xuống 82%.

## 5. Thử thách cá nhân

Theo SEED **5103**, đề được giao là **Thử thách #4 — Embedding**: chạy `knn_predict` với k = 1, 5, 25, báo F1 và giải thích vì sao k quá nhỏ hoặc quá lớn đều có hại.

**Cách làm:** dùng lại embedding MiniLM đã tính cho tập train (`E_tr`) và test (`E_te`). Với mỗi tin test, `knn_predict` lấy k tin train gần nhất theo cosine (`top_k_similar`), đoán spam nếu hơn nửa số hàng xóm là spam. Chạy trên toàn bộ 775 tin test với k = 1, 5, 25 rồi tính precision, recall, F1.

**Kết quả:**

| k | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 1 | 0.865 | 0.947 | 0.905 |
| 5 | 0.873 | 0.937 | 0.904 |
| 25 | 0.864 | 0.937 | 0.899 |

F1 gần như không đổi (chênh ~0.006, chỉ vài tin trên 95 spam) nên coi như hòa.

**Giải thích:**
- **k quá nhỏ:** kết quả chỉ phụ thuộc vào 1 hàng xóm nên rất nhạy với nhiễu. Nếu hàng xóm đó bị gán nhãn sai hoặc là tin lạ thì dự đoán sai và không có hàng xóm khác kéo lại (overfit).
- **k quá lớn:** hàng xóm lấn ra ngoài vùng cục bộ, mất chi tiết cục bộ. Vì lớp ham chiếm ~87%, phiếu bầu dễ bị ham lấn át, ở cực đoan k = n thì luôn đoán lớp đa số, giống mô hình lười.
- Trong lần chạy này cả ba giá trị k đều cho F1 xấp xỉ nhau, nên chưa thấy rõ cái hại ở hai đầu. Nếu cần chọn k thì nên chọn bằng tập val, không dùng tập test.
- Mình đã đoán trước k = 25 làm precision tăng, thực tế precision gần như đứng yên (0.864).

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (~116 SMS/giây trung bình), p95 < 50 ms/tin.

**Chọn phương án: router hai tầng.**
- **Tầng 1:** Encoder (MiniLM) + MLP head (3c) xử lý tin có xác suất chắc chắn (p < 0.3 hoặc p > 0.7).
- **Tầng 2:** tin lưỡng lự (p ∈ [0.3, 0.7]) gửi cho Haiku.

**Số liệu dẫn:**
- **Độ trễ:** 3c có p95 **32.5 ms**, đạt SLA. Haiku p95 **1000 ms**, Opus **1590 ms**, kNN **57.96 ms**: cả ba đều trượt SLA nên không thể dùng cho toàn bộ luồng.
- **Chất lượng:** 3c có F1 test **0.963**, cao nhất trong các cách đạt SLA. BoW + LogReg và BoW + MLP rất nhanh (0.09 và 0.34 ms) nhưng F1 trên 50 case khó chỉ **0.359 và 0.439**, trong khi Haiku đạt **0.969**.
- **Chi phí:** Nếu dùng Haiku cho toàn bộ: 300 triệu tin/tháng × $0.0101 / 1k tin ≈ **$3,000/tháng**. Opus ≈ **$120,800/tháng**. Nếu chỉ ~5% tin lưỡng lự (giả định, chưa đo trên val) thì Haiku còn khoảng **$150/tháng**.
- **Throughput:** 3c đo được khoảng 32 tin/s ở chế độ cả lô, nên cần ít nhất 116 / 32 ≈ 4 instance cho tải trung bình (suy ra từ số đo, chưa tính giờ cao điểm).

**Đánh đổi chấp nhận:**
- Tin lưỡng lự đi qua Haiku mất khoảng 1 giây, vượt SLA 50 ms. Mình chấp nhận xử lý riêng nhóm này (bất đồng bộ hoặc giữ tin lại) và chỉ cam kết p95 < 50 ms cho phần đi qua tầng 1.
- Router không cứu được trường hợp mô hình rẻ sai nhưng rất tự tin, và tỉ lệ 5% chưa được đo.
- Chi phí "0 đồng" của 3c chỉ tính tiền gọi API, vẫn có tiền hạ tầng (khoảng 4 instance trở lên).
- Cần thêm chuẩn hóa đầu vào và lọc câu ra lệnh cho AI trước khi đưa vào LLM, vì prompt phòng thủ một mình không cải thiện injection.
- Số liệu LLM chỉ đo trên 40 tin và bộ 55 case viết tay, nên cần đo lại trên dữ liệu thật trước khi triển khai.

## 7. AI log và điều đã học

| Phần | Mình hỏi AI gì | AI trả lời/viết gì | Mình sửa / kiểm chứng gì |
|---|---|---|---|
| Khái niệm mở đầu | Mô hình lười là gì; câu trả lời về data leakage có đúng không; accuracy của mô hình luôn đoán ham | Giải thích 3 nghĩa của "lười"; chỉ ra mình viết sai "overlifting" (phải là overfitting) và leakage làm điểm test bị thổi phồng; baseline 4516/5169 ≈ 0.8737 | Sửa lại thuật ngữ; ghi `baseline_accuracy` = 0.87 và đạt kiểm tra |
| BoW + LogReg | Dự đoán F1; giải thích gradient của "free"; chọn tin bị đoán sai; thử lr = 0.001 và 50; tìm 2 lỗi trong đoạn code "AI viết" | Gợi ý F1 ≈ 0.90; đoán lr = 50 sẽ zigzag; chỉ ra 2 lỗi (đánh giá trên tập train, vocab dựng từ toàn bộ dữ liệu) | Chạy thật: F1 0.903. AI đoán sai lr = 50 (loss vọt lên rồi hội tụ về 0.007), mình ghi nhận phần dự đoán sai |
| MLP | Đếm tham số; dự đoán val loss; bỏ ReLU thì F1 ra sao | 128,129 tham số; dự đoán bỏ ReLU về ~LogReg | In số tham số từ notebook; chạy thật thì bỏ ReLU vẫn F1 0.935, AI đoán sai |
| Embedding | Dự đoán tìm kiếm ngữ nghĩa, 6 tin tricky, bảng 3a/3b/3c, kNN k = 1/5/25 | Đoán BoW chỉ đúng ~3/6; giải thích transfer learning; viết lời giải thích về k | Thực tế BoW đúng 5/6. AI từng đưa số suy ra ("bắt 90/95 spam") nên mình yêu cầu bỏ, chỉ dùng số trong bảng. F1 k = 25 mình lấy 0.899 từ output notebook |
| LLM | Cấu hình provider và API key; viết `call_openai`; prompt phân loại; `parse_label`; temperature/max_tokens | Hướng dẫn cấu hình, sửa lỗi `NotImplementedError`; chỉ ra "not spam" bị tính thành spam và câu trả lời rỗng bị tính thành ham | Chọn Anthropic; mock test và gọi thử đều đạt; few-shot làm F1 giảm trái dự đoán |
| Pareto và SLA | Cách nào nằm trên đường Pareto, cách nào vượt SLA | Dự đoán 3c nằm trên đường Pareto | Hình thật cho thấy 3c bị BoW + MLP lấn át; mình sửa lại nhận xét theo p95 và F1 SUB |
| 50 case khó | Dự đoán accuracy, prompt injection, tiền xử lý, độ đại diện; nghĩ thêm 5 case | Đoán LLM nhỏ dễ bị injection; chấm câu trả lời của mình về độ đại diện bộ test | Thực tế Haiku đúng 89% nhóm injection (sai 1/9), không phải yếu như đoán; mình chạy lại với 55 case |
| Sản phẩm thật | Chọn phương án, chi phí LLM mỗi tháng, thiết kế router | Tính chi phí từ đơn giá trong bảng; đề xuất router | Kiểm tra đơn giá lấy từ bảng của notebook (không bịa); chỉnh "chi phí 0 đồng" thành còn tiền hạ tầng |

### Điều hiểu ra hôm nay

Mình hiểu rằng điểm cao trên tập test thường chưa nói lên model dùng được ngoài đời: BoW + MLP có F1 SUB 0.923 nhưng chỉ còn 0.439 khi gặp spam né từ khóa. Mình cũng hiểu rằng với tập nhỏ (SUB 40 tin, 6 tin tricky, 55 case) chênh vài điểm F1 chỉ là vài tin, nên không thể xếp hạng model chỉ bằng con số. Cuối cùng, chọn model phải cân cả F1, độ trễ và chi phí cùng lúc, vì cách tốt nhất ở một mặt thường kém ở mặt khác (Opus chính xác nhất nhưng chậm gấp ~50 lần và đắt gấp 40 lần Haiku).
