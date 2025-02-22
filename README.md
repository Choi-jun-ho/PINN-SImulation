<!-- Improved compatibility of back to top link: See: https://github.com/Choi-jun-ho/PINN-SImulation/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the PINN-SImulation. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">PINN-SImulation</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/Choi-jun-ho/PINN-SImulation"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Choi-jun-ho/PINN-SImulation">View Demo</a>
    ·
    <a href="https://github.com/Choi-jun-ho/PINN-SImulation/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/Choi-jun-ho/PINN-SImulation/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

기존 인공지능은 물리현상을 학습하여 재현하기가 매우 어려웠습니다.
저희는 https://doi.org/10.1016/j.jcp.2018.10.045 해당 논문을 참고하여 아파트 화재에 대해서 $CO_2, CO, O_2, Tempretue, Velocity, Velocity_x, Velocity_y, Velocity_z$를 실시간 예측을 하는 것을 목표로 합니다.




<p align="right">(<a href="#readme-top">back to top</a>)</p>




### Built With


* [![Python][Python.com]][Python-url]
* [![Pytorch][Pytorch.com]][Pytorch-url]
* vscode
* numpy
* deepxde
* matplotlib
* pandas
* tqdm
* numba

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

이는 로컬에서 프로젝트를 설정하는 방법에 대한 지침을 제공할 수 있는 방법의 예입니다.
로컬 복사본을 실행하려면 다음의 간단한 예제 단계를 따르십시오.

### Prerequisites

다음은 소프트웨어를 사용하는 데 필요한 항목을 나열하는 방법과 설치 방법의 예입니다.

* conda
  ```sh
  conda env create -f ai_pinn.yaml
  ```

### Installation

1. [FDS(Fire Dynamics Simulator)](https://pages.nist.gov/fds-smv/downloads.html) 설치
    - bundle 설치
2. ./apartment_multi 폴더에서 Apartment_place2.fds 실행
3. ./apartment_room 폴더에서 Apartment_multi_case.fds 실행
4. git cloe
   ```sh
   git clone https://github.com/Choi-jun-ho/PINN-SImulation.git
   ```
5. vscode ctrl+shift+p >Jupyter: Export Current Python File as Jupyter Notebook
6. 실행관련된 주석 제거후 순차적 실행


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

작성 중...

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- 작성중..

See the [open issues](https://github.com/Choi-jun-ho/PINN-SImulation/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

- 현재는 계획 없습니다.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Coding Convention

### Commit Convention
- Feat : 새로운 기능에 대한 커밋
- Modify: 내용 수정
- Remove: 내용 삭제 (폴더나 파일)
- Dir: 폴더 및 파일 구조 변경
- Fix : 버그 수정에 대한 커밋
- Build : 빌드 관련 파일 수정에 대한 커밋
- Chore : 그 외 자잘한 수정에 대한 커밋
- Ci : CI 관련 설정 수정에 대한 커밋
- Docs : 문서 수정에 대한 커밋
- Style : 코드 스타일 혹은 포맷 등에 관한 커밋
- Refactor : 코드 리팩토링에 대한 커밋
- Test : 테스트 코드 수정에 대한 커밋
- Init: 시스템 초기 설정

### Example

> <제목>
Feat: 관심지역 알림 ON/OFF 기능 추가(#123)
> 
> 
> 
> <본문>
> 커밋 메시지 본문으로 "왜", "무엇을 위해", "어떻게" 변경했는지와 상세 내용 추가 설명하기
> 
> <꼬릿말>
> 
> - Fixes: 이슈 수정중 (아직 해결되지 않은 경우)
> - Resolves: 이슈를 해결했을 때 사용
> - Ref: 참고할 이슈가 있을 때 사용
> - Related to: 해당 커밋에 관련된 이슈번호 (아직 해결되지 않은 경우)

#### Branch Convention

- feture, develop, release, hotfix,main

<!-- LICENSE -->
## License

<!-- Distributed under the MIT License. See `LICENSE.txt` for more information. -->
작성중...

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Junho Choi - junho74589@gmail.com

Project Link: [https://github.com/Choi-jun-ho/PINN-SImulation](https://github.com/Choi-jun-ho/PINN-SImulation)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Choi-jun-ho/PINN-SImulation.svg?style=for-the-badge
[contributors-url]: https://github.com/Choi-jun-ho/PINN-SImulation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Choi-jun-ho/PINN-SImulation.svg?style=for-the-badge
[forks-url]: https://github.com/Choi-jun-ho/PINN-SImulation/network/members
[stars-shield]: https://img.shields.io/github/stars/Choi-jun-ho/PINN-SImulation.svg?style=for-the-badge
[stars-url]: https://github.com/Choi-jun-ho/PINN-SImulation/stargazers
[issues-shield]: https://img.shields.io/github/issues/Choi-jun-ho/PINN-SImulation.svg?style=for-the-badge
[issues-url]: https://github.com/Choi-jun-ho/PINN-SImulation/issues
[license-shield]: https://img.shields.io/github/license/Choi-jun-ho/PINN-SImulation.svg?style=for-the-badge
[license-url]: https://github.com/Choi-jun-ho/PINN-SImulation/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/%EC%A4%80%ED%98%B8-%EC%B5%9C-58b64323a/
[Python.com]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white
[Python-url]: https://www.python.org/
[Pytorch.com]: https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white
[Pytorch-url]: https://pytorch.org/