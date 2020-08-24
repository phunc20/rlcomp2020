Dear các đội thi,

Trong phạm vi bài này, BTC xin cung cấp và gợi mở một số thông tin tới các đội chơi. Hy vọng những thông tin dưới đây sẽ giúp các đội chơi đưa ra được những Agent sử dụng phương pháp học tăng cường hiểu quả, và có kết quả cao trong cuộc thi.

 

1. Chiến lược trong cuộc thi.

Trong cuộc thi, BTC hướng các đội chơi tới 3 mục tiêu sau:

- Tìm đường: các đội chơi cần phải huấn luyện cho Agent sao có thể tìm được đường đi đến các vị trí vàng.

- Giảm thiểu số năng lượng bị trừ: trên đường đi đến các vị trí vàng, Agent cần phải được huấn luyện để có thể né được những vật cản (có thể là cây, bẫy, hoặc đầm lầy) với mục tiêu giảm thiểu số năng lượng bị trừ.

- Đối kháng: mục tiêu này được thể hiện khi các đội chơi cần phải thi đấu với 03 bots của BTC trong vòng loại, và thi đấu với những đội chơi khác ở các vòng trong. Do đó, mỗi đội chơi cần phải huấn luyện được cho Agent có khả năng dự đoán được chiến thuật của những đội chơi khác trong lúc thi đấu. Ví dụ, Agent của đội chơi có thể dự đoán được các agent khác đang hướng tới những vị trí vàng nào, từ đó Agent của đội chơi có thể chọn được một hướng phù hợp. Một ví dụ khác nữa đó là đối với đầm lầy, Agent nào vào trước sẽ có lợi thế khi bị trừ ít năng lượng hơn các Agent vào sau. Do đó, Agent của đội chơi nên có ưu tiên đi tới sớm hơn những vị trí vàng có giá trị bị bao bọc bởi đầm lầy.

 

2. Huấn luyện một Agent sử dụng phương pháp học tăng cường (Reinforcement Learning)

Trong cuộc thi, để huấn luyện một Agent theo phương pháp học tăng cường, các đội chơi có thể tiếp cận từ những thuật toán cổ điển như Q-learning hay SARSA sử dụng dạng bảng (tabular) để lưu không gian trạng thái (state space) và hành động (action), đến những thuật toán học tăng cường có tích hợp các mạng học sâu (deep neural networks) hay còn gọi là học tăng cường sâu (Deep reinforcement learning). Đối với những thuật toán cổ điển, điểm hạn chế đó là khi không gian trạng thái lớn, việc lưu dưới dạng bảng sẽ dẫn tới không đủ bộ nhớ và tăng thời gian tìm kiếm. Bên cạnh đó, việc định nghĩa không gian trạng thái sẽ tốn nhiều công sức trong những bài toán phức tạp như sử dụng thông tin ảnh. Với hướng tiếp cận học tăng cường sâu (DRL), những điểm hạn chế trên được giải quyết thông qua việc sử dụng những mạng học sâu (deep neural networks) để ước lượng không gian trạng thái, từ đó giảm thiểu được kích thước cũng như thời gian tìm kiếm khi chỉ phải lưu cấu hình và tham số của mạng. Ngoài ra, những mạng học sâu như Convolutional Neural Network/CNN còn cho phép đưa toàn bộ thông tin ảnh như một không gian trạng thái. Việc này cho phép những thuật toán DRL làm việc trực tiếp trên dữ liệu thô giúp giảm thiểu công sức trong việc định nghĩa không gian trạng thái.

 

Với những thế mạnh của hướng tiếp cận DRL, BTC đã cung cấp Code mẫu cho phép huấn luyện một Agent sử dụng thuật toán Deep Q-learning (DQN), và một trong 03 Bots của BTC cũng được huấn luyện bằng DQN sử dụng một mạng học sâu có 02 lớp ẩn là Fully connected, và một số thông tin cơ bản như thông tin của Agent và vàng được sử dụng thay vì đưa cả bản đồ vào không gian trạng thái. Tuy nhiên, Bot DQN được BTC xây dựng ở mức cơ bản, các đội chơi có thể hướng tiếp cận tới những mạng học sâu phức tạp hơn có sử dụng CNN để có thể học trên toàn bộ không gian bản đồ. Trong quá trình huấn luyện, các đội chơi nên chú ý tới vấn đề đảm bảo Agent tham dò đầy đủ không gian trạng thái thông qua tham số epsilon và epsilon decay. Nếu tham số epsilon giảm nhanh và không gian trạng thái lớn, thăm dò của Agent sẽ không được đầy đủ, và dẫn đến việc Agent học không hiệu quả. Ngược lại, tham số epsilon giảm chậm và không gian trạng thái nhỏ, Agent sẽ tốn nhiều thời gian thăm dò, và đồng nghĩa việc thời gian huấn luyện tăng lên. Khi đó, các đội chơi sẽ có ít thời gian thử nghiệm các tham số cũng như những thuật toán khác.

Bên cạnh đó, các đội chơi có thể tạo ra những Bot với chiến thuật khác nhau để đưa vào huấn luyện cùng với Agent. Cách tiếp cận trên sẽ giúp Agent của đội chơi học và thích nghi được với đa dạng các chiến thuật khác nhau tạo khả năng chiến thăng cao hơn.

Đối với Vòng Bảng, các đội chơi phải đấu với 03 Bots của BTC, được cập nhật KHÔNG GIỚI HẠN số phiên bản Agent, nhưng chỉ được thi đấu tối đa 20 lần với Bot của Ban tổ chức (Trận đấu với bot sẽ được tự động kích hoạt sau khi code của đội thi được xác thực thành công). Các đội thi hãy test cẩn thận ở máy local trước khi đưa code lên để tránh trường hợp bị lãng phí lượt nộp bài!

 

Hướng dẫn cách upload code: https://youtu.be/V23ijfCkFI4
