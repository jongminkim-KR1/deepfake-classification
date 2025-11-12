# GCP 인스턴스와 VS Code 원격 연결하기

### GCP 인스턴스 생성

1. 구글 계정으로 GCP 계정 생성

- 계정 결제 메뉴에서 활성화 필요

2. 프로젝트 설정

![1.jpg](images/1.jpg)

![2.jpg](images/2.jpg)

![3.jpg](images/3.jpg)

3. 인스턴스 설정

![4.jpg](images/4.jpg)

- GPU 유형

![5.jpg](images/5.jpg)

NVIDIA V100 또는 L4 추천 → 요청 승인 필요

- 머신 유형

![6.jpg](images/6.jpg)

![7.jpg](images/7.jpg)

n1-standard-4 또는 n1-standard-8 추천

- 부팅 디스크

![8.jpg](images/8.jpg)

![9.jpg](images/9.jpg)

cuda가 포함된 디스크 선택

학습 데이터가 큰 경우 100GB이상 추천

- 네트워킹

![10.jpg](images/10.jpg)

![11.jpg](images/11.jpg)

### VS Code 연결

1. SSH 연결

- VS Code에 설치

![13.jpg](images/13.jpg)

- ssh 키 생성

```python
ssh-keygen -t rsa
```

명령 프롬프트에 입력

해당 경로(예: C:\Users\사용자\.ssh 폴더) id_000, id_000.pub 파일 생성 → .pub 파일에 있는 키 복사

![14.jpg](images/14.jpg)

- GCP에 ssh 키 등록

![15.jpg](images/15.jpg)

2. VS Code와 ssh 연결

vs code에서 ctrl + shift + p 누르고 Remote-SSH: Add New SSH Host... 선택

![16.jpg](images/16.jpg)

![17.jpg](images/17.jpg)

![18.jpg](images/18.jpg)

복사해서 VS Code에 붙여넣기

![19.jpg](images/19.jpg)

로컬 파일 선택

![20.jpg](images/20.jpg)

연결 확인 & 화살표 클릭하여 연결

![21.jpg](images/21.jpg)

리눅스 선택 후 continue 선택 → 시작

###### 참고: https://ariz1623.tistory.com/371, https://ariz1623.tistory.com/372