# Báo cáo cá nhân --- Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

-   **MSSV:** 2A202604115
-   **Họ và tên:** Nguyễn Văn Lâm
-   **SEED:** 5103

> Ghi chú: Báo cáo được tổng hợp từ `day01_lab.ipynb` và
> `lab1_report.json` đã cung cấp. Những mục notebook chưa có kết quả
> chạy được ghi rõ là chưa có dữ liệu, không tự tạo số liệu thay thế.

## 1. Bằng chứng day01_lab

  -----------------------------------------------------------------------
  Tệp                     Có?                     Ghi chú
  ----------------------- ----------------------- -----------------------
  `day01_lab.ipynb`       ☑                       Đã cung cấp notebook;
                                                  các output và kết quả
                                                  kiểm tra được lưu trong
                                                  file

  `lab1_report.json`      ☑                       Đã cung cấp; SEED 5103,
                                                  kết quả và trạng thái
                                                  checks

  `hand_calc.jpg`         ☑                       Đã cung cấp
                                                  trong các tệp đính kèm
  -----------------------------------------------------------------------

Notebook ghi nhận **15/15 kiểm tra đạt**, gồm vectorize,
sigmoid/predict_proba, train_logreg, metrics, build_mlp, train_mlp,
cosine/top_k_similar, knn_predict, prompt/parse_label,
call_openai/llm_classify_batch và 5 câu tính tay.

## 2. Kết quả chính

Bảng dưới đây chép các số liệu thực tế có trong `lab1_report.json`. Các
model LLM không xuất hiện trong phần `results` của file báo cáo, nên
không có số liệu để điền.

  ------------------------------------------------------------------------
  Mô hình             F1 test      F1 SUB 40   p95 (ms/tin)     USD/1k tin
  ------------ -------------- -------------- -------------- --------------
  BoW + LogReg          0.903          0.824          1.042          0.000

  BoW + MLP             0.935          0.923          0.353          0.000

  MiniLM + kNN          0.904          0.889         65.061          0.000

  MiniLM +              0.947          0.919         34.489          0.000
  LogReg                                                    

  MiniLM + MLP          0.963          0.919        108.993          0.000
  head                                                      
  ------------------------------------------------------------------------

Nhận xét từ số liệu đã lưu: - MiniLM + MLP head có F1 test cao nhất
trong các mô hình đã có kết quả: **0.963**. - MiniLM + LogReg có F1 test
**0.947**, p95 **34.489 ms/tin**. - BoW + MLP có F1 test **0.935**, p95
**0.353 ms/tin**, throughput **23,490.701 tin/giây**. - MiniLM + kNN và
MiniLM + MLP head không đạt ngưỡng p95 dưới 50 ms trong lần đo này. -
Chi phí USD/1k tin của các mô hình ML/DL/Embedding được ghi là 0 vì đây
là các mô hình chạy local trong bài lab, không phải chi phí API.

## 3. Dự đoán và thực tế

### 3.1. BoW + Logistic Regression

-   **Dự đoán:** F1 khoảng 0.89; precision nhỉnh hơn recall vì từ khóa
    spam trong dữ liệu khá đặc trưng.
-   **Thực tế:** F1 test **0.903**; precision **0.9875**, recall
    **0.8316**.
-   **Đánh giá:** Dự đoán khá sát. F1 thực tế cao hơn dự đoán khoảng
    0.013. Precision cao hơn recall đúng như dự đoán; mô hình ít gắn
    nhầm ham thành spam nhưng vẫn bỏ sót một phần spam.

### 3.2. MLP và Logistic Regression

-   **Dự đoán:** Validation loss sẽ giảm rồi đi ngang/tăng nhẹ; MLP gần
    bằng Logistic Regression vì cả hai vẫn dùng feature Bag-of-Words.
-   **Thực tế:** F1 test của MLP là **0.935**, cao hơn Logistic
    Regression **0.903** khoảng 0.032.
-   **Đánh giá:** Dự đoán về việc MLP gần bằng Logistic Regression chưa
    hoàn toàn khớp; MLP tốt hơn trong lần chạy này. Notebook không lưu
    đủ output biểu đồ/epoch để xác định chính xác epoch bắt đầu overfit.

### 3.3. Embedding

-   **Dự đoán:** Encoder + LogReg sẽ xử lý tốt hơn BoW với tin nhắn có
    từ khóa gây nhiễu hoặc spam né từ khóa.
-   **Thực tế:** MiniLM + LogReg đạt F1 test **0.947**, cao hơn BoW +
    LogReg (**0.903**). Tuy nhiên, F1 SUB của MiniLM + LogReg là
    **0.919**, cao hơn BoW + LogReg (**0.824**).
-   **Đánh giá:** Kết quả tổng thể ủng hộ nhận định embedding có lợi thế
    về biểu diễn ngữ nghĩa. Tuy nhiên, notebook không lưu output dự đoán
    từng tin trong thí nghiệm 3.B, nên chưa thể kết luận chính xác model
    nào đúng ở từng case tricky.

### 3.4. Gemini và Few-shot

-   **Dự đoán:** Gemini 3.8 Flash có F1 cao hơn Gemini 3.5 Flash-Lite
    khoảng 1--3%; model lớn tốn chi phí hơn. Few-shot với 4 ví dụ có thể
    tăng F1 nhưng làm tăng input token.
-   **Thực tế:** Notebook có ghi nhận gọi thử Gemini Flash-Lite thành
    công và mock test `call_openai`/`llm_classify_batch` đạt. Tuy nhiên,
    `lab1_report.json` không có kết quả 4.A/4.B của các model Gemini.
-   **Đánh giá:** Chưa đủ dữ liệu để xác nhận hoặc bác bỏ dự đoán về F1
    và chi phí. Cần chạy lại hai cell thí nghiệm 4.A và 4.B để lấy kết
    quả thực tế.

## 4. Phân tích 50 case khó

Notebook có định nghĩa bộ HARD_EVAL gồm 50 case thuộc 7 nhóm: - Ham có
vẻ giống spam: 10 case. - Spam giọng tự nhiên: 10 case. - Thao túng/lừa
đảo: 8 case. - Spam né từ khóa: 6 case. - Prompt injection: 6 case. -
Tin giao dịch hợp lệ: 6 case. - Nói về spam: 4 case.

chưa làm

## 5. Thử thách cá nhân

Theo SEED **5103**, đề được giao là **Thử thách #4 --- Embedding**:

> Chạy `knn_predict` với k = 1, 5, 25. Báo F1. Giải thích vì sao k quá
> nhỏ hoặc quá lớn đều có hại.

Chưa làm


## 6. Phương án cho sản phẩm thật

Yêu cầu: lọc 10 triệu SMS/ngày, tương đương khoảng 116 SMS/giây trung
bình, với p95 dưới 50 ms/tin.

Từ benchmark đã lưu:

-   **BoW + LogReg:** p95 1.042 ms, throughput 7,549.692 tin/giây, F1
    test 0.903. Đạt yêu cầu độ trễ và throughput trong lần đo.
-   **BoW + MLP:** p95 0.353 ms, throughput 23,490.701 tin/giây, F1 test
    0.935. Đây là phương án cân bằng tốt nhất trong các mô hình đã đo về
    F1, p95 và throughput.
-   **MiniLM + LogReg:** p95 34.489 ms, đạt ngưỡng p95 nhưng throughput
    đo được chỉ 30.447 tin/giây, thấp hơn mức trung bình 116 tin/giây.
    Cần tối ưu/batching hoặc mở rộng tài nguyên trước khi triển khai ở
    quy mô yêu cầu.
-   **MiniLM + kNN:** p95 65.061 ms, không đạt ngưỡng p95.
-   **MiniLM + MLP head:** p95 108.993 ms, không đạt ngưỡng p95.

**Lựa chọn đề xuất:** dùng BoW + MLP làm lớp phân loại chính vì F1 test
đạt 0.935, p95 rất thấp và throughput đo được cao hơn đáng kể mức 116
tin/giây. Có thể dùng MiniLM + LogReg làm lớp kiểm tra bổ sung cho những
SMS có xác suất dự đoán lưỡng lự, sau khi tối ưu throughput và kiểm thử
tải.

**Đánh đổi:** BoW/MLP có thể kém linh hoạt hơn embedding hoặc LLM với
spam né từ khóa và ngữ cảnh phức tạp. MiniLM cải thiện khả năng hiểu ngữ
nghĩa nhưng tăng độ trễ và chi phí tài nguyên. LLM có thể dành cho các
trường hợp khó thay vì gọi cho toàn bộ 10 triệu tin mỗi ngày.

Chưa thể tính chi phí tháng khi dùng LLM từ báo cáo hiện tại vì không có
kết quả USD/1k tin của Gemini trong `lab1_report.json`. Không nên lấy
con số dự đoán thay cho chi phí thực nghiệm.

## 7. AI log và điều đã học

  -----------------------------------------------------------------------------------------------------
  Phần              Đã hỏi AI               AI hỗ trợ                          Kiểm chứng / chỉnh sửa
  ----------------- ----------------------- ---------------------------------- ------------------------
  Cấu hình LLM      Cách chuyển giữa        Hướng dẫn cấu hình `PROVIDERS`,    Chọn Gemini; kiểm tra
  provider          Gemini, OpenAI, Groq,   `base_url`, API key, model và tham gọi thử Gemini
                    OpenRouter và Ollama    số token                           Flash-Lite thành công

  Hàm gọi API       Cách dùng chung         Hướng dẫn viết                     Chạy mock test cho
                    `call_openai()` cho các `provider_kwargs()`,               `call_openai()` và
                    provider                `call_openai()` và `call_llm()`    `llm_classify_batch()`
                    OpenAI-compatible                                          

  Lỗi khi chạy      Lỗi                     Xác định hàm vẫn còn               Thay hàm và chạy lại
  notebook          `NotImplementedError`   `raise NotImplementedError`, hướng cell kiểm tra
                    trong `call_openai()`   dẫn thay bằng phần triển khai thật 

  Prompt phân loại  Cách tạo prompt         Hướng dẫn                          Các check liên quan
                    zero-shot/few-shot và   `build_classification_prompt()`,   prompt/parse và API mock
                    chuyển nhãn trả về      `parse_label()` và                 đều đạt
                                            `llm_classify_batch()`             

  Phân tích kết quả Cách diễn giải F1, chi  Hỗ trợ viết nhận định dự đoán      Phân biệt dự đoán với
                    phí và ảnh hưởng của 4  trước thực nghiệm                  kết quả đo; chưa có số
                    ví dụ few-shot                                             liệu Gemini trong báo
                                                                               cáo cuối
  -----------------------------------------------------------------------------------------------------

### Điều hiểu ra sau bài lab

Hôm nay mình hiểu rõ hơn rằng accuracy không đủ để đánh giá bộ lọc spam;
cần nhìn thêm precision, recall và F1 vì hai loại lỗi chặn nhầm tin hợp
lệ và bỏ sót spam có tác động khác nhau. Mình cũng hiểu mỗi cách biểu
diễn văn bản có điểm mạnh riêng: BoW nhanh nhưng phụ thuộc từ khóa, còn
embedding biểu diễn ngữ nghĩa tốt hơn nhưng tốn thời gian xử lý. Với
LLM, prompt và few-shot có thể cải thiện cách phân loại nhưng cũng làm
tăng token đầu vào, chi phí và độ trễ, nên cần đo thực tế trước khi chọn
triển khai.

------------------------------------------------------------------------

