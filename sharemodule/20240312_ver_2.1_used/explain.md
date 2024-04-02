# 2024
## 01
### 11
train.py: get_output 에서 iterator 의 결과값으로 pred 및 b_label 만 받아서 lossfunction 에 넣는 것으로 조정

lossfunction의 baseline 을 형성

sharemodules 에 lossfunction.py 추가

lossfunction 관련 함수는 trainutils.py 가 아닌 lossfunction 에 입력하는 것으로 변경

lossfunction 의 입력값은 pred, b_label 로 고정

