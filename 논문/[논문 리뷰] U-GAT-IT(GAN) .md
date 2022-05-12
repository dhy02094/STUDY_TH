# [논문 리뷰] U-GAT-IT(GAN)

# 0.

KT 에이블스쿨에서 마무리 과정으로 빅프로젝트를 진행하고 있다.

여기서 내가 주로 맡은 분야는 논문 ugatit을 이용해 사용자가 이용할 수 있게 api화 하고 그걸 django를 이용해 웹 페이지를 만들어 배포하는 것이였다.

이 글에서는 ugatit을 직접 학습 시키는데 1Epoch당 227시간이 걸려 할 수 없었지만 그 이론이 기존 cycle gan과 다르게 재밌고 질문 대비를 위해 부족하지만 최대한 공부한 만큼 남겨 보려한다.

이후 어떻게 flask를 이용해 api로 만들었고 어떻게 그것을 이용해 장고에서 배포했는지도 빠른 시일내에 남기겠다.

# 1. Introduction

cycle Gan, UGATIT 같은 GAN 방식은 소위 unsupervised image-to-image translation이라고 한다.

- unsupervised image-to-image translation : 정답이 주어지지 않고 연관이 없는 두 이미지를 학습 시키는 것이다. 즉 이 논문에서 처럼 사진 → 애니메이션 // 사진 → 풍경화 로 만드는 필터 같은 느낌이다.

이전 cycle GAN에서 성공적으로 위 내용을 성공했지만 실제로 성능이 좋지 못했다.

그것을 보완할 재밌는 방법을 추가했다. 바로 **CAM(class activation map), attention 이다.**

밑에서 다시 자세히 설명하겠다.

그 다음 중요한 것이 AdaLIN이다.

여기는 새로운 Normalization function을 사용했다. 이는 변화된 결과에 상당한 영향을 미친다고 한다.

간단히 말하면 ADAIN + LN 이라고 한다. 결국 이 함수는 어텐션 모델이 적절하게 변화의 양을 제어할 수 있게 도와준다는데 그 결과 모델의 구조가 하이퍼 파라미터의 조정 없이 Image translation이 가능 하다고 한다.

밑에서 한번 더 짚고 넘어가겠다.

결국 이 다음으로는 특별한 것 없이 기존 cycle GAN과 비슷하게 Source domain $X_s$ 의 이미지들을 target domain $X_t$로  function $G_{s->t}$ 를 학습 시키는 것이 목표이다.

논문에서는 두개의 generators $G_{s->t}$, $G_{t->s}$ 와 두 개의 discriminators $D_s, D_t$로 구성되어 있다. 밑에 사진에서 자세히 설명하겠다.

# 2. attention + model

논문의 구조와는 좀 다르게 중요한 부분 순서로 가보겠다.

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FvXlZl%2FbtqYbIBtsmR%2FOtEJweMsF5kfVltONPDBJ0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FvXlZl%2FbtqYbIBtsmR%2FOtEJweMsF5kfVltONPDBJ0%2Fimg.png)

다음은 논문 구조이다. 

## Generator

generator는 말 그대로 이미지를 생성하는 부분이다. 기본적으로 autoencoder랑 전체적인 틀이 같다.

일단 이미지의 특징을 뽑아 featuremap을 만드는 Encoder → Decoder(이미지의 특징으로 재생성하는) 구조를 가지고 있다.

자세히 하나하나 뜯어보면 encoder 과정에서는 이미지의 특징을 살리는 downsampling 과정과 Residual Blocks 과정을 거치고 있다. 

- 여기서 Residual Blocks: weight layer를 통과한 F(x)와 통과하지 않은 기존 x의 합이다. 즉 (F(x)+x) 이다. 이것을 하는 이유는 바로!! vanishing gradient 때문이다. 기존 학습을 시키면서 계속된 미분으로 0이 되면서 기울기가 사라질수 있는데 계속 x를 더해줌으로 기울기소실을 방지해주는 것이다.

그 다음 아주 중요한 **attention이 나온다.**

바로 auxilary classifier이다. 이것은 중간에서 softmax 하는 것과 비슷하다고 보면된다. 아무튼 이것을 함으로써 값 x가 sourcedomain에서 왔을 확률을 계산해준다. 

조금 더 풀어보면 encoder에서 얻은 Featuremap과 그 피쳐맵에서 중요하게 본 영역에 가중치 w를 곱해서 decoder의 input으로 보내주는 역할을 하는 것이다.

수식으로 다시 보자.

![Untitled](%5B%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%20%E1%84%85%E1%85%B5%E1%84%87%E1%85%B2%5D%20U-GAT-IT(GAN)%201abcccf3ec4c42fcad9e24920d093c6b/Untitled.png)

이렇게 해서 큰 값이 나온 feature 위주로 보게 되는것이다. 해당 논문에서 수많은 결과들을 봤을 때 눈 변환이 가장 심했다. 따라서 아마 눈이 가장 큰 값이 였을 것이다.

이후 Decorder 과정에서는 attention map의 fully connected layer에서 구한 $\gamma, \beta$ 에 의해 계산된다. 

그 다음 AdaLIN으로 정규화 시킨다.

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcuxHwo%2FbtqYiDloKBp%2Fs5kbzNB6kmsLOudW0lxba1%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcuxHwo%2FbtqYiDloKBp%2Fs5kbzNB6kmsLOudW0lxba1%2Fimg.png)

여기서 $\mu_I, \mu_L$  그리고 $\sigma_I, \sigma_L$은 각각 channel-wise, layer-wise 평균과 표준 편차이다. 

나머지 $\rho$는 optimizer에 의해 결정되는 gradient, $\tau$는 learning rate를 뜻한다.

너무 어려워서 깊게는 못 들어가겟지만 간략하게 블로그들을 보며 이해한 내용으로는 Layer normalization은 각 feature map의 통계량이 같다고 가정하고, instance normalization은 다르다고 가정한다. 따라서 전자는 원래 이미지의 내용이 달라가는 경향이 있고 후자는 잘 유지하는 경향이 있다고 한다. 

위 식에서 보면 AdaLIN의 앞의 식 channel-wise의 관련한 식은 instance normalization으로 계산된것이고 뒤는 layer-wise로 계산된 layer normalization으로 계산 된 값으로 위에 전자와 후자를 적절하게 섞어서 사용해 decoder 과정을 마친다.

## Discriminator

이제 추격자 discriminator를 파보자.

generator에 비해 별거 없다 똑같이 encoder과정으로 이미지의 특징을 추출하고 이것을 attention 방식을 사용해서 attention featuremap을 똑같이 뽑아낸다. 근데 이제는 이 discriminator는 사진이 target에서 온건지 domain에서 온건지 구분하는 Classifier 과정으로 넘어간다. 다른 gan은 16진수의 값이 나온다고 하는데 특이하게 여기는 그냥 binary classifer로 이진수의 값이 나온다.

# 3. LOSS

이 모델의 loss는 4가지의 함수가 섞여있다. 마지막 한 가지 빼고는 기존 cycle GAN과 동일하다.

### 1) Adversarial loss

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbytr0O%2FbtqYhyxXqBs%2FpAgbK3iMUTb0GaGK6GC9A0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbytr0O%2FbtqYhyxXqBs%2FpAgbK3iMUTb0GaGK6GC9A0%2Fimg.png)

- Least Squares GAN 로스식이라고 한다.

### 2) Cycle loss

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdBK4PA%2FbtqX6f7L04M%2FqO8Z9zDH3GD67mgIrkQli0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdBK4PA%2FbtqX6f7L04M%2FqO8Z9zDH3GD67mgIrkQli0%2Fimg.png)

- 계속 똑같은 이미지를 만들어내는 Mode collapse문제를 완하시키기위해 적용했다고 한다.
- 이미지 X가 주어졌을 때, $X_s$→$X_t$→$X_s$ 로 순차적으로 돌고 이미지는 원래의 도메인으로 변환된다.

### 3) Identity loss

Input 이미지와 Output 이미지의 컬러가 균형을 맞출수 있도록 다음 식을 적용했다.

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcTuAgl%2FbtqYgHvbtPK%2FKk94KeIR00zUHLNe5P39k0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcTuAgl%2FbtqYgHvbtPK%2FKk94KeIR00zUHLNe5P39k0%2Fimg.png)

### 4) CAM loss

이게 이 논문에서 들어간 식이다. 

이름처럼 개선해야하는 위치와 현재 상태에서 두 개의 도메인이 어떤 차이를 갖는지 확인해주는 로스 이다.

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcJG6SJ%2FbtqYkaXEtmf%2Fb6sDzeFvKx09zOQpcXtXJ0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcJG6SJ%2FbtqYkaXEtmf%2Fb6sDzeFvKx09zOQpcXtXJ0%2Fimg.png)

### 마무리 : Full objective

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcLhJ5g%2FbtqYecCnMOg%2FfWpoPlEdgcJSKIw1CMXNA1%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcLhJ5g%2FbtqYecCnMOg%2FfWpoPlEdgcJSKIw1CMXNA1%2Fimg.png)

결국 위 로스를 모두 다 합쳐서 최적화 시켜서 위에 있는 모든 모델 구조를 다 학습시킨다.

# 4. Experiments

조금 죄송하긴 한데 .... 데이터셋과 result에 대해선 간단히 남기도록 하겠다.

학습시킨 논문 데이터셋은 kaggle에 Selfie2anime 라는 데이터이고

결과는 다음과 같다

[https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FX21h2%2FbtqYxo2usdL%2FrN8SK7FOhNLM3hYXhmBJt0%2Fimg.png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FX21h2%2FbtqYxo2usdL%2FrN8SK7FOhNLM3hYXhmBJt0%2Fimg.png)

다음에서 보면 알수 있듯이 확실히 cam이 들어갔다고 압도적으로 성능이 좋아졌다.

# 마무리

처음 써보는 논문리뷰고... 다른 블로거들을 많이 참고해서 많이 부족하고 가독성도 떨어지겠지만 점점 더 발전해가면서 다음에 더 실력이 성장하면 누구나 잘 이해하고 도움을 얻어 갈 수 있는 논문 리뷰를 작성하겠다.

![620FA3FD-6541-449F-AA58-5EDF33CF86A8.png](%5B%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%20%E1%84%85%E1%85%B5%E1%84%87%E1%85%B2%5D%20U-GAT-IT(GAN)%201abcccf3ec4c42fcad9e24920d093c6b/620FA3FD-6541-449F-AA58-5EDF33CF86A8.png)

ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ

적용해본 내얼굴이다.