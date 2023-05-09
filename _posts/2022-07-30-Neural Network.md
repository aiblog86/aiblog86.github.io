---
title: Neural Network
author: nttrinh
date: 2022-07-30 18:30:00 +0700
categories: [Blogging, Deep Learning]
tags: [Deep Learning]
pin: true
---


# Neural Network - Xu hướng công nghệ AI/ML

## I. Giới thiệu

### 1.1. Bài toán phân loại và các thuật toán AI/ML cổ điển
**Bài toán phân loại**
Trong khoảng từ 5 đến 10 năm trở lại đây, khi nhắc về các ứng dụng thực tiễn liên qua đến lĩnh vực AI/ML thì bài toán phân loại là bài toán cơ bản được biết đến và áp dụng rộng rãi nhất. Đặc biệt là trong y học hiện đại, ngày nay, người ta đang dần ứng dụng các kĩ thuật của AI/ML để dự đoán khả năng bị ung thư của một người từ đó sớm đưa ra phương hướng ngăn chặn.

![](https://i.imgur.com/DEwkiCy.png) 

Để nói về bài toán phân loại, tôi tạm phân nó thành 2 loại khác nhau dựa trên **đặc trưng (feature) đầu vào của dữ liệu**:
1. Các đặc trưng độc lập với nhau, và thường thì giá trị mỗi đặc trưng đều nằm trong một miền cụ thể.
2. Các đặc trưng trong cùng một lớp (class) có cùng mối liên hệ nào đó với nhau - tuyến tính hoặc phi tuyến*. 

Sau đây là hai ví dụ cụ thể của hai dạng trên và các thuật toán tương ứng:
1. Dữ liệu có các đặc trưng độc lập với nhau. Ví dụ sơ khai nhất của dạng này đó chính là xác định xem một email có phải là spam hay không. Bằng cách thống kê kiểm tra xem một email spam thường thường tồn tại "các từ hoặc cụm từ nào" để phân loại. Thuật toán điển hình cho dạng này là Decision tree, Navie Bayes.
2. Các đặc trưng trong cùng một lớp có cùng mối liên hệ nào đó với nhau. Điển hình của dạng bài toán này trong thực tế là khả năng vượt qua bài kiểm tra của một học sinh dựa trên số giờ ôn tập. Ở đây, ta có thể đặt câu hỏi là: "Không phải một học sinh ôn càng nhiều thì kết quả đậu sẽ càng cao hay sao? Vậy ta chỉ cần kiểm tra xem số giờ ôn tập của một sinh viên có lớn hơn hoặc bằng một số gì đó thì đậu và người lại. Vậy thì nó khác gì dạng đầu tiên?" Tuy nhiên thực tế chứng minh nó không phải luôn như vậy (các bạn có thể tham khảo trên [Wiki](https://en.wikipedia.org/wiki/Logistic_regression)). Thuật toán điển hình cho dạng này là Logistic Regression, Softmax và sau này là Neural Network.

<img src='https://i.imgur.com/6Jw4JEh.png' width="500px">


### 1.2. Các vấn đề của thuật toán phân loại cổ điển và Neural Network
Ở phần này tôi sẽ chỉ tập trung vào dạng thứ 2 của bài toán phân loại vì nó liên quan đến Neural Network.
#### Logistic Regression
Mặc dù tên của thuật toán thuộc dạng hồi quy (regression), tuy nhiên nó lại được sử dụng trong bài toán phân loại. Thuật toán logistic thực hiện hai bước sau để  xây dựng mô hình và phân loại một đối tượng:
1. Giả định mối liên hệ giữa các đặc trưng bằng phương trình $Z = WX^T + b$, trong đó $W$ là trọng số của từng đặc trưng, $X$ là đặc trưng đầu vào và $b$ là tham số bias.
2. Cho giá trị $Z$ đi qua hàm sigmoid, ta được một giá trị ***theta($\theta$)*** nằm trong khoảng $(0,1]$, nếu $\theta \leq0.5$ ta cho nó thuộc lớp A, ngược lại nó thuộc lớp B.

Trong quá trình **"huấn luyện""** mô hình, trọng số $W$ sẽ được cập nhật liên tục đến khi hội tụ. Còn khi **"dự đoán"**, ta sẽ cố định trọng số $W$ và $b$, ta chỉ đặt giá trị đặc trưng đầu vào $X$ vào trong phương trình $Z$ sau đó tiến qua bước 2. Đối với mô hình Logistic Regression người ta huấn luyện và cập nhật trọng số bằng hàm [BCE](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) loss.

<img src='https://i.imgur.com/1LWBzul.png' width='500px'>

#### Softmax Regression
Đối với thuật toán Logistic Regression, tồn tại một điểm yếu cố hữu là nó **rất khó** để áp dụng cho các bài toán phân loại trên 2 lớp. Sau đó thuật toán Softmax Regression ra đời để giải quyết vấn đề của thuật toán Logistic Regression bằng công thức.
<img src='https://i.imgur.com/oGLZy9Q.png' width='350px'>


Với công thức trên Softmax Regression thỏa mãn 2 yếu tố cho bài toán phân lớp:
1. Xác suất luôn nằm trong khoảng $(0,1]$.
2. Tổng các xác suất luôn bằng $1$.

#### Neural Network
Mặc dù softmax đã giải quyết được vấn đề phân loại cho nhiều lớp nhưng cả hai thuật toán trên đều mắc phải một khuyết điểm vô cùng lớn đó là xem mối liên hệ giữa các đặc trưng là một mối **liên hệ tuyến tính**. Tuy nhiên trong thực tế mối liên hệ giữa các đặc trưng thường là mối **liên hệ phi tuyến** (ví dụ như giá trị từng pixel trong ảnh), rất phức tạp để có thể xác định. Đó là lí do cấu trúc mạng ***Neural Network*** ra đời để giải quyết vấn đề trên. 

![](https://i.imgur.com/CoE2O5m.png)

Ở phần sau tôi sẽ lí giải **cấu trúc** của mạng Neural Network và cách nó **giải quyết vấn đề mối liên hệ phi tuyến giữa các đặc trưng**.

## II. Phương pháp
### 2.1. Từ bộ não sinh học đến mạng Neural nhân tạo
Có thể bạn không biết, nhưng mạng Neural Network thật chất được lấy ý tưởng từ chính bộ não sinh học của chúng ta. Trên phương diện sinh học, bộ não được cấu thành từ hàng triệu các neural thần kinh (mỗi neural thực hiện một nhiệm vụ khác nhau) kết nối với nhau tạo thành một mạng lưới. Từ đó tạo nên một hệ thống phức tạp giúp con người có khả năng suy nghĩ và phát triển vượt bậc.
### 2.2. Cấu trúc cơ bản của một mạng Neural
#### Cấu trúc cơ bản của một mạng Neural Network
Về mặt cấu trúc, một mạng Neural Network có 3 phần chính: 
1. ***Đặc trưng đầu vào*** (input layer). Đây là lớp nhận các đặc trưng dữ liệu đầu vào của "từng đối tượng".
2. ***Các lớp ẩn*** (hidden layer) được chồng lên nhau. Chức năng chính của lớp này tìm ra mối liên hệ phi tuyến của đặc trưng đầu vào.
3. ***Kết quả đầu ra*** (output layer). Đối với bài toán phân loại thì mỗi node tượng trưng cho một lớp (class) và xác suất đối tượng thuộc về lớp đó.

Trong mỗi lớp có thể chứa một hoặc nhiều nút (node hoặc perceptron), mỗi node mang một tính năng riêng tùy thuộc vào hàm kích hoạt (activation function) mà ta cài đặt cho nó như là Sigmoid, ReLU, Softmax, ... (phần này tôi sẽ đề cập ở phần [2.3](https://hackmd.io/ORTc3kkDQSmBi5R9z1-o1A?both&fbclid=IwAR1FQF1mTiFvUdVExA3zRzOgpolKtL7w3OcKPTR0If5qjNfaepZwnPfA5Pc#23-Node-trong-m%E1%BA%A1ng-Neural)), chính vì vậy mà mạng Neural Network còn có tên khác là Multilayer Perceptrons.

### 2.3. Node trong mạng Neural
#### Node
Mỗi node (trừ lớp input) gồm 3 thành phần chính:
1. Vector đặc trưng đầu vào - $X (x_1, x_2, ..., x_n)$.
2. Vector trọng số - $W(w_1, w_2, ..., w_n)$.
3. Kết quả đầu ra.
![](https://i.imgur.com/rogpihh.png)

Nếu để ý, bạn có thể dễ dàng nhận ra mỗi node trong mạng Neural Network chính là một thuật toán thu nhỏ thuộc dạng 2* mà tôi đề cập ở trên. Như vậy, giá trị đầu ra của mỗi node được xác định như sau:
1. Tính giá trị $Z$ bằng công thức $Z = WX^T + b$. 
2. Sau đó giá trị $Z$ được đưa qua hàm kích hoạt để trả về kết quả cuối cùng. 

### 2.3. Hàm kích hoạt
#### Nhiệm vụ của hàm kích hoạt
Dựa vào cách tính kết quả đầu ra của từng node, ta có thể thấy nhiệm vụ chính của hàm kích hoạt là biến đổi giá trị đầu vào của từng node thành một dạng khác.
#### Các hàm kích hoạt thường dùng
##### Sigmoid
Công thức:
<img src="https://www.gstatic.com/education/formulas2/397133473/en/sigmoid_function.svg" width="250px">
**Ý nghĩa:** Sigmoid function thật chất là thuật toán Logistic Regression, về mặt toán học sigmoid là một hàm phi tuyến. Người ta thường xuyên sử dụng nó ở lớp output dành cho các bài toán phân loại nhị phân.
##### Softmax
Công thức:
<img src="https://www.gstatic.com/education/formulas2/397133473/en/softmax_function.svg" width="250px">

**Ý nghĩa:** tương tự như Sigmoid function, về mặt toán học softmax cũng là một hàm phi tuyến và cũng được sử dụng ở lớp output. Tuy nhiên sự khác biệt của nó với sigmoid là nó được dùng cho bài toán phân loại có số lượng lớp lớn hơn 2.
##### ReLu
Công thức:
<img src="https://i.imgur.com/z98gBY1.png" width="450px">

**Ý nghĩa:** có lẽ đây là hàm kích hoạt được sử dụng nhiều nhất trong các lớp hidden layer vì nó tránh được hiện tượng [vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem), một hiện tượng xảy ra không mong muốn trong quá trình huấn luyện mô hình. Đồng thời về mặt toán học nó cũng là một hàm phi tuyến.

### 2.4. Cách hoạt động của mạng Neural Network
Vậy là ở phần trên tôi đã giới thiệu cho các bạn các thành phần cơ bản nhất của một mạng Neural Network. Ở phần này, tôi sẽ giới thiệu về cách hoạt động để các bạn có một góc nhìn tổng quan nhất để có thể tiếp tục tìm hiểu sâu hơn. Ở đây tôi sẽ chia ra thành 2 quá trình là huấn luyện và dự đoán để dễ theo dõi
#### Quá trình huấn luyện
Quá trình huấn luyện của Neural Network cũng tương tự như quá trình huấn luyện của thuật toán Logistic Regression mà tôi đã nói ở trên. Tuy nhiên, nó sẽ phức tạp hơn đôi chút và được chia thành 2 phần: **Feed Forward** và **Back-propagation**
##### Feed Forward
Dữ liệu của chúng ta sẽ đi qua tuần tự sau: Input layer $\rightarrow$ Hidden layer $\rightarrow$ Output layer. Kết quả ở lớp output (ta sẽ lấy node có giá trị xác suất cao nhất là kết quả đầu ra) sẽ được dùng để tính toán hàm loss.
##### Back-propagation
Đây là bước khó nhất khi mới tiếp cận với mạng Neural Network. Ở đây sau khi đã tính loss ở bước feedforward. Ta sẽ đi cập nhật các trọng số $W$ bằng 2 kỹ thuật là [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent) và [Chain Rule](https://en.wikipedia.org/wiki/Chain_rule) (đây là bước khác biệt so với thuật toán Logistic Regression).
Thực hiện vòng lặp feedforward và backpropagation cho tới khi giá trị loss của chúng ta "hội tụ".
#### Quá trình dự đoán
Tương tự như mô hình Logistic Regression, ở bước dự đoán ta sẽ cố định các trọng số W, sau đó cho dữ liệu đi qua mạng Neural Network và ra kết cuối cùng.

### 2.5 Liệu một node hay một layer có đủ để giải quyết vấn đề phức tạp?
Câu trả lời đơn giản là không, tôi sẽ đưa ra ví dụ để làm rõ hơn vấn đề này. Giả sử bạn có một bộ dữ liệu phân bố như sau:

![](https://i.imgur.com/nanAUGK.png)

Ta có thể thấy rõ, với bộ dữ liệu trên ta không thể tìm ra một đường thẳng để chia đôi dữ liệu thành 2 phần rõ rệt. Nếu chỉ có một node bạn sẽ nhận được mô hình phân loại như sau:

![](https://i.imgur.com/Wd326Fv.png)

Vậy nếu ta tăng số lượng node lên gấp đôi, từ 1 lên 2 thì sao? Đây là kết quả:

![](https://i.imgur.com/pHEprGH.png)

Ta có thể thấy kết quả đã tốt hơn nhưng vẫn chưa đủ, ta lại tiếp tục tăng số node lên gấp đôi.

![](https://i.imgur.com/GdeQCZK.png)

Ở đây bài toán tôi đưa ra vẫn còn khá đơn giản nên bạn chỉ cần một lớp là đã có thể giải quyết vấn đề. Tuy nhiên, với các bài toán phức tạp hơn ta sẽ cần thêm nhiều lớp hidden layer và số lượng node ở mỗi lớp để giải quyết. Các bạn có thể tham khảo đoạn code cho bài toán phân loại tôi vừa làm ở [đây](https://colab.research.google.com/drive/1ZARQNiQAVkBezOMf7uALwZTCjIahomZO?usp=sharing). 


## III. Cài đặt chương trình
### 3.1. Đọc dữ liệu
Tải dữ liệu MNIST có sẵn từ PyTorch
```python=
train_data = torchvision.datasets.MNIST(
                        root = 'data',
                        train = True,                         
                        transform = transforms.ToTensor(),
                        download = True,
)

test_data = torchvision.datasets.MNIST(
                        root = 'data', 
                        train = False, 
                        transform = transforms.ToTensor(),
)
```
Một số hình ảnh về dữ liệu.
<img src="https://i.imgur.com/mSJaD3V.png" width="500">

Để đưa dữ liêu vào mô hình ta cần tạo DataLoader của từng tập train và test.
```python=
train_loader = utils.data.DataLoader(train_data, 
                                  batch_size=256, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count()
)

test_loader = utils.data.DataLoader(test_data, 
                                  batch_size=256, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count()
)
```
### 3.2. Thiết kế mô hình
Vì để so sánh công bằng cho các mô hình, tôi đã train Logistic Regression, Softmax Regression và Neural Network theo cách như sau:
*    30 epochs đầu: `learning_rate`: 0.001
*    10 epochs tiếp theo: `learning_rate`: 0.0025
*    10 epochs cuối: `learning_rate`: 0.0025 và `momentum`: 0.9

Và hàm optimizer chung cho cả ba mô hình là **Stochastic Gradient Descent** ([`SGD`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)).
```python
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=learning_rate, 
                            momentum=momentum)
```
#### 3.1. Logistic Regression

```python=  
class LogisticRegression(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.sigmoid(outputs)
        return outputs
    
log_reg = LogisticRegression(in_features=784, out_features=10)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(log_reg.parameters(), 
                            lr=learning_rate, 
                            momentum=momentum)
```
Tuy nhiên là PyTorch đã hỗ trợ cho chúng ta một hàm loss là [`BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) và đã bao gồm sẵn lớp Sigmoid. Theo tác giả, điều này sẽ giúp cho việc tính toán ổn định hơn bằng cách kết hợp việc tính toán này vô cùng một lớp, thay vì tính Sigmoid được đầu ra và đưa qua hàm tính loss.
```python=
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(log_reg.parameters(), 
                            lr=learning_rate, 
                            momentum=momentum)
```

#### 3.2. Softmax Regression
```python=
class SoftmaxRegression(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
    
    def forward(self, inputs):
        outputs = self.linear(inputs)
        return outputs
    
softmax_reg = SoftmaxRegression(in_features=784, out_features=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(softmax_reg.parameters(), 
                            lr=learning_rate, 
                            momentum=momentum)
```
#### 3.3. Neural Networks
```python=
class NeuralNetwork(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(in_features, 128, bias),
                        nn.ReLU(),
                        nn.Linear(128, 64, bias),
                        nn.ReLU(),
                        nn.Linear(64, out_features, bias)
        )
        
    def forward(self, inputs):
        return self.model(inputs)

model = NeuralNetwork(in_features=784, out_features=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=learning_rate, 
                            momentum=momentum)
```
### Đánh giá và trực quan hóa kết quả
Bảng kết quả trên bộ dữ liệu test của từng mô hình.

| **Model**               | **Test Accuracy** |
|-------------------------|-------------------|
| **Logistic Regression** | 0.8461            |
| **Softmax Regression**  | 0.9057            |
| **Neural Networks**     | **0.9273**        |

Một số hình ảnh về quá trình train.
![](https://i.imgur.com/GUMJ4IS.png)

## IV. Kết luận
Vậy là tôi đã giới thiệu cho các bạn những khái niệm cơ bản nhất về cấu trúc cũng như cách vận hành của một mạng Neural Network. Sau đây là những kiến thức tôi muốn các bạn lắng động lại và **bỏ túi mang về:**
1. Một mạng Neural Network cơ bản gồm 3 phần: Input layer, Hidden layer và Output layer.
2. Mỗi node trong mạng đều thực hiện một nhiệm vụ riêng, được quyết định bởi các trọng số $W$ và hàm kích hoạt của nó.
3. Mỗi hàm kích hoạt đều có một tác dụng riêng và ta phải linh hoạt để ứng dụng.
4. Tùy thuộc vào yêu cầu của từng bài toán ta sẽ thiết kế một cấu trúc mạng phù hợp cho nó. Tuy nhiên hãy nhớ, một mô hình càng phức tạp sẽ yêu cầu càng nhiều dữ liệu để quá trình huấn luyện hội tụ.

Mặc dù đã hình thành và phát triển từ rất lâu nhưng nó chỉ mới nở rộ trong khoảng 10 năm trở lại nhờ sự phát triển của các hệ thống phần cứng máy tính. Hiện tại, nó đã có rất nhiều biến thể và được ứng dụng trong nhiều lĩnh vưc, trong đó hai phát triển nhất là lĩnh vực Thị giác Máy tính (CV) và Xử lí Ngôn ngữ Tự nhiên (NLP).

## V. Bài tập
Để củng cố kiến thức về Neural Network ở phía trên, chúng mình đề xuất với các bạn một bài tập với yêu cầu như sau:
1. Hãy sử dụng kiến trúc Neural Network để phân loại bộ dữ liệu [Fashion MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).
2. Các bạn có thể tìm hiểu và sử dụng các phương pháp nâng cao hơn như *mạng neural tích chập* (Convolutional Neural Network - CNN), thay đổi hàm tối ưu (optimizer) và tinh chỉnh các tham số như `learning_rate`, `batch_size`, ... nhằm tối ưu mô hình cũng như hiểu rộng hơn về các mô hình neural network.

### Code tham khảo
Dưới đây là code tham khảo được build trên [PyTorch](https://pytorch.org) với kiến trúc tương tự như minh họa phía trên.
**`import` các thư viện cần thiết**
```python=
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
```
**Lựa chọn acceleration cho mô hình**
```python=
device = ('cuda' if torch.cuda.is_available() else 'cpu')
```
**Đọc dữ liệu và xử lý dữ liệu**
```python=
train_data = torchvision.datasets.FashionMNIST(
                            root = 'data',
                            train = True,                         
                            transform = transforms.ToTensor(),
                            download = True,
)

test_data = torchvision.datasets.FashionMNIST(
                            root = 'data', 
                            train = False, 
                            transform = transforms.ToTensor(),
)

# Data visualization
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```
**Chia data thành từng batch và tạo `DataLoader` để đưa dữ liệu vào mô hình**
```python=
# You can adjust number of batches to fit your memory here
batch_size = 256
# Create DataLoader
train_loader = utils.data.DataLoader(train_data, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count()
)

test_loader = utils.data.DataLoader(test_data, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=os.cpu_count()
)
```
**Định nghĩa mô hình**
Các bạn có thể tham khảo một số mô hình CNN [tại đây](https://pytorch.org/vision/stable/models.html).
```python=
class NeuralNetwork(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(in_features, 128, bias),
                        nn.ReLU(),
                        nn.Linear(128, 64, bias),
                        nn.ReLU(),
                        nn.Linear(64, out_features, bias)
        )
        
    def forward(self, inputs):
        return self.model(inputs)
    
# Model initialization
model = NeuralNetwork(in_features=784, out_features=10)
```
**Định nghĩa hàm `loss` và hàm `optimizer`**

```python=
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
```
**Định nghĩa hàm `train` cho mô hình**
```python=
def train(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, -1)
            running_loss += loss.item() / len(dataloaders)
            correct_preds += torch.sum(preds == labels.data).float().item()
            running_acc = correct_preds / len(dataloaders.dataset)
        print(f"Loss: {running_loss}, accuracy: {running_acc}")
    return model
```
**Train mô hình**
```python=
model = fit(model, train_loader, criterion, optimizer, N_EPOCHS)
```
**Thực nghiệm trên bộ test**
```python=
model.eval()
correct_preds = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        
        outputs = nn.Softmax(dim=-1)(outputs)
        _, preds = torch.max(outputs, -1)
        correct_preds += torch.sum(preds == labels.data).float().item()
        
print(f"Accuracy: {correct_preds/len(test_loader.dataset)}")
```
## Tham khảo
[1] BCELoss — PyTorch 1.11.0 documentation. PyTorch. https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
[2] P.T. BCEWithLogitsLoss — PyTorch 1.11.0 documentation. PyTorch. https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
[3] P.T. SGD — PyTorch 1.11.0 documentation. PyTorch. https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
[4] Wikipedia contributors. (2022a, March 19). Chain rule. Wikipedia. https://en.wikipedia.org/wiki/Chain_rule
[5] Wikipedia contributors. (2022b, April 4). Gradient descent. Wikipedia. https://en.wikipedia.org/wiki/Gradient_descent
[6] Wikipedia contributors. (2022c, April 8). Vanishing gradient problem. Wikipedia. https://en.wikipedia.org/wiki/Vanishing_gradient_problem
[7] Wikipedia contributors. (2022d, April 10). Logistic regression. Wikipedia. https://en.wikipedia.org/wiki/Logistic_regression

**Nguyễn Trung Tuấn/19522477
Trịnh Nhật Tân/19522179**