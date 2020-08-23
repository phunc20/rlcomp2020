Dear các đội thi,

Dưới đây là bài tổng hợp về các kĩ thuật trong việc cải thiện model DQN, được chia sẻ bởi VNPT_IC Team (Team đang dẫn đầu trên bảng xếp hạng)

BTC mong rằng bài viết sẽ giúp ích cho các team khác trong việc training model.

 

🌈 Double DQN (https://arxiv.org/pdf/1509.06461.pdf) : Thay vì sử dụng 1 mạng neural chung để select action và tính Q-value ta sẽ sử dụng 2 mạng neural, 1 mạng để select action và 1 mạng để tính Q-value. 2 mạng neural này chung 1 kiến trúc, chỉ khác nhau về weight.

🌈 Dueling Network (https://arxiv.org/pdf/1511.06581.pdf) : Output mạng neural thay vì chỉ đưa ra giá trị Q(s,a) thì đưa ra 2 giá trị V(s) và A(s,a). Sau đó kết hợp 2 giá trị này để tính ra Q(s,a).

 

🌈 Prioritized Experience Replay (https://arxiv.org/pdf/1511.05952.pdf) : 1 cải tiến cho vấn đề sampling replay. Với model DQN trước đây ta chỉ random ngẫu nhiên các replay từ buffer. Ý tưởng của PER là thay vì lấy random ta sẽ ưu tiên lấy nhưng replay chưa được sử dụng và những replay có TD-error cao.

 

🌈 N-Step Learning : Thay vì sử dụng 1 step để tính Q target, giá trị reward của n-step liên tiếp sẽ được sử dụng để tính Q target.

 

🌈 Distributed Agents: Thay vì chỉ sử dụng 1 actor để generate sample ta sẽ dùng multi actors. Các actor này chạy song song trên nhiều luồng cùng lúc. Kĩ thuật này không những giúp việc huấn luyện nhanh hơn mà còn giúp tăng đáng kể độ chính xác. Với việc đặt cho mỗi actors 1 hệ số epsilon khác nhau ta còn có thể khiến mô hình khám phá ra nhiều nước đi mới hơn. Hầu hết các kiến trúc DQN state-of-the-art hiện nay đều đang là Distributed Agents (Ape-x, R2D2, NGU, Agent57, Muzero, ….)

 

🌈 Short term memory (https://openreview.net/pdf?id=r1lyTjAqYX): Với những bài toán yếu tố sequence đóng vai trò quan trọng, như lịch sử của các state trước có ảnh hưởng nhiều đến những state sau thì đây là 1 phương pháp khá hiệu quả.

 

🌈 Muzero (https://arxiv.org/pdf/1911.08265.pdf) : Đây là 1 mô hình cải tiến từ AlphaZero. Nếu như AlphaZero chỉ có thể áp dụng được trên những game cờ thì giờ đây Muzero có đã có thể áp dụng thuật toán tương tự trên tất cả các game khác. Về thuật toán tìm kiếm Monte carlo tree search thì không có gì thay đổi so với AlphaZero. Điểm khác biết là Muzero có thêm 2 mạng neural, 1 mạng để predict reward của môi trường trả về, 1 mạng để tái hiện lại trạng thái của môi trường ở những state tiếp theo. Nhờ đó Muzero có thể áp dụng thuật toán tìm kiếm cây MTCC mà không cần phải sử dụng đến môi trường trả về, nghĩa là mô hình sẽ tự học ra một game của riêng nó. Muzero hiện đang đạt SOTA trên hầu hết các game atari hiện nay.

 

🌈 Agent 57 (https://arxiv.org/pdf/2003.13350.pdf) : Đây cũng là một model khá nổi tiếng, mới được Google Deepmind public gần đây, model đạt kết quả trên human performance trên toàn bộ 57 games Atari. Model là sự kết hợp của rất nhiều kĩ thuật cải tiến của DQN. Trong đó cải tiến đáng kể nhất là kĩ thuật “intrinsic motivation rewards” và “meta-controller”. 2 kĩ thuật này giúp model giải quyết vấn đề khá nan giải trong DQN là exploration (việc khám phá các nước đi mới mặc dù có thể không tốt bằng các nước đi cùng thời điểm nhưng đem lại reward tốt hơn trong các nước đi tương lai) và time horizon (những bài toán mà reward chỉ đạt được sau rất nhiều state)

 

Chúc các đội thi thành công!
