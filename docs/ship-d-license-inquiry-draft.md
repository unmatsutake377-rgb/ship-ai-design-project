# Ship-D 라이선스 문의 초안 (발송 대기 — 오너 승인 필요)

상태: **초안**. 발송은 오너 계정으로 오너가 직접 (또는 승인 후).
발송처 후보: ① GitHub 이슈 (github.com/noahbagz/ShipD — 공개 게시)
② 저자 이메일 (논문 교신저자 주소).

## 배경 (우리 상황 요약)

- 원저장소에 LICENSE 파일 없음 = 법적 기본값은 "모든 권리 유보"
  (열람은 가능하나 재배포·파생물 배포 권리는 명시가 없음)
- 우리 사용 형태: 로컬 실행 전용 (코드 경로 주입 import, 복사 없음),
  데이터 재배포 없음, 생성 선체 STL·렌더는 저장소 커밋·공개 금지 중
- 문의 목적: ① 이 사용 형태의 확인 ② 파생 형상(우리 파이프라인이
  Ship-D 파라미터로 생성한 선체) 공개 가능 여부 ③ 올바른 인용

## 영문 초안 (GitHub 이슈용)

제목: `License clarification for ShipD dataset and code`

---

Hi, thank you for open-sourcing ShipD — it has been a fantastic
learning resource.

I am an undergraduate student (naval architecture) building an
educational, open-source ship design pipeline
(https://github.com/unmatsutake377-rgb/ship-ai-design-project).
We use ShipD as follows:

- The dataset and `HullParameterization` code are used **locally
  only** — imported at runtime from a local copy, never copied into
  or redistributed with our repository.
- Hull meshes generated from ShipD parameter vectors are kept
  local; we have not published any generated STL files or renders,
  pending license clarification.

Since the repository currently has no LICENSE file, could you
clarify:

1. Is our local, non-redistributing use acceptable to you?
2. May we publish **derived artifacts** (STL meshes / renders of
   hulls generated from ShipD parameter vectors, possibly modified
   by our optimizer) in our public repository and documentation?
3. Do you plan to add an explicit license (e.g. MIT / CC-BY)?
4. What is your preferred citation? (We currently cite the ShipD
   paper by Bagazinski & Ahmed.)

We are happy to follow whatever terms you prefer — thank you!

---

## 발송 후 처리 관례

- 답변 오면: docs/superpowers/specs/2026-08-05-shipgen-generator-design.md
  §1 라이선스 자세 갱신 + 허용 범위에 따라 커밋 금지 규칙 완화/유지
- 무응답 시: 현행 보수 자세 유지 (로컬 전용, 공개 금지)
