# 제출 계획: baseline 스모크 (2026-10-27 02:42)

- 현황: brl1 down. 빈 CPU brl0 32, brl5 56, brl4 12, brl2 8, brl3 4. 빈 메모리 노드당 약 28–64G. 빈 GPU brl2 5, brl3 6, brl5 4, brl0 1. brl 대기 53개. 내 job 없음.
- 목적: 실제 크기(n0=8000, n1=400, 7600행)에서 smoten, smote_enc(CPU)와 ctgan(GPU)이 도는지, 시간이 얼마인지 측정. 본실행 자원 산정의 근거.
- 자원: CPU job 4 CPU, 16G, 1시간. GPU job GPU 1, 4 CPU, 16G, 1시간(기록 없음, 작은 표라 여유 있게).
- GPU/CPU: smoten·smote_enc는 sklearn 계열이라 CPU. ctgan은 torch라 GPU.
