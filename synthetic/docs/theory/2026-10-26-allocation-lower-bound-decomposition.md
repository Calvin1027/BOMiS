# SZ 배분: 하한 명제와 오차 분해

작성 2026-10-26. 대상 초안 `refs/draft_certifying_structural_zeros.pdf`(텍스트 `refs/draft_certifying_structural_zeros.txt`).

## 0. 목적과 결론

초안의 정리 3은 sampling zero(SZ)에 놓인 총질량 $\Gamma(\hat q) = \hat q(S_1 \setminus O)$가 $M_0$에 가깝다는 것을 보장한다. SZ 셀 사이에서 질량을 어떻게 나눴는지는 다루지 않는다(초안 452줄). 이 문서는 가정 1·2와 정리 1~3을 바꾸지 않고 두 가지를 더한다.

- **명제 A (배분 하한).** 가정 1·2의 클래스 $\mathcal M_k(r)$ 안에서는 어떤 oversampler도 SZ 사이 배분 오차를 균일하게 작게 만들 수 없다. $p_0$를 정확히 알고 $S_1$을 정확히 알아도 마찬가지다.
- **명제 B (배분 오차 분해).** BOMiS의 배분 오차는 (가) 후보 집합에서 빠진 SZ 질량, (나) 디코더가 후보 집합 위에서 $p_1$의 모양을 틀린 정도, (다) 선택과 같은 수의 사본이 디코더 법칙에서 벗어난 정도, 세 항의 합 이하이다. 이 부등식은 아무 가정 없이 성립한다.

합성 실험에서는 $p_1$을 알므로 세 항을 모두 정확히 계산할 수 있다.

advisor 검토(2026-10-26)를 반영했다. 명제 A는 증명이 맞다는 판정을 받았다. 명제 B는 사건 $H$, 실효 후보 집합, 기댓값 형태, 분해의 해석을 고쳤다. 명제 A는 "안 보인 셀의 질량은 식별되지 않는다"는 사실만 쓰고 가정 1의 국소 구조를 쓰지 않으므로 자명하다는 지적을 받을 수 있다. 논문에는 짧은 remark나 부록으로 넣는 편이 안전하다는 의견이 있었다(배치는 미정).

## 1. 기호와 배분 오차

초안 3.1절의 기호를 따른다. $U = \prod_j C_j$, 클래스 $y$의 법칙 $p_y$와 support $S_y$, $D_0 \sim p_0^{n_0}$, $D_1 \sim p_1^{n_1}$은 서로 독립이다. $O$, $O_0$는 $D_1$, $D_0$에 나온 서로 다른 셀의 집합이다. 조각 $\beta$, 실현 가능 조각 집합 $F_1$, $\pi_y(\beta) = p_y(\beta)$, 클래스 $\mathcal M_k(r)$은 초안과 같다.

$$\mathrm{SZ} = S_1 \setminus O, \qquad M_0 = p_1(\mathrm{SZ}), \qquad P^\star = p_1(\cdot \mid \mathrm{SZ}) \quad (M_0 > 0).$$

oversampler는 $(D_0, D_1, \xi)$에 대해 가측인 확률질량함수 $\hat q$를 낸다. $\xi$는 데이터와 독립인 보조 난수이다.

**정의 (배분 오차).**
$$\mathcal A(\hat q) = \begin{cases} \mathrm{TV}\big(\hat q(\cdot \mid \mathrm{SZ}),\ P^\star\big) & M_0 > 0,\ \hat q(\mathrm{SZ}) > 0 \\ 1 & M_0 > 0,\ \hat q(\mathrm{SZ}) = 0 \\ 0 & M_0 = 0 \end{cases}$$

$\mathrm{TV}(P, Q) = \tfrac12 \sum_c |P(c) - Q(c)|$이다. $\mathcal A$는 총질량 $\hat q(\mathrm{SZ})$와 무관하다. 총질량은 정리 3의 $|\Gamma - M_0|$가 따로 다룬다. ADEMP 설계 7.2절 (b)의 $\mathrm{TV}(\hat q(\cdot \mid \mathrm{SZ}), p_1(\cdot \mid \mathrm{SZ}))$가 이 양이다.

## 2. 명제 A (배분 하한)

### 2.1 설정

$(p_0, p_1)$이 가정 1을 만족한다고 하자. $S_1$ 안에 서로 겹치지 않는 셀 쌍 $\{c_j, c_j'\}$, $j = 1, \dots, J$를 잡는다($2J$개 셀이 모두 다르다).

$$a_j = p_1(c_j), \quad b_j = p_1(c_j'), \quad m_j = \min(a_j, b_j), \quad \eta \in (0, 1).$$

조각 $\beta$에 대해 $J(\beta) = \{ j : \beta$가 $c_j, c_j'$ 중 정확히 하나를 포함$\}$로 둔다.

**조건 (S).** 모든 $\beta \in F_1$에서
$$\pi_1(\beta) - r\,\pi_0(\beta) \ \ge\ \eta \sum_{j \in J(\beta)} m_j .$$

**법칙족.** $\omega \in \{0,1\}^J$, $s_j = 2\omega_j - 1 \in \{-1, +1\}$에 대해
$$p_1^\omega(c_j) = a_j + s_j \eta m_j, \qquad p_1^\omega(c_j') = b_j - s_j \eta m_j,$$
쌍에 속하지 않는 셀에서는 $p_1^\omega = p_1$이다. $p_0$는 모든 $\omega$에서 같다.

### 2.2 명제

**명제 A.** 위 설정에서 다음이 성립한다.

(i) 모든 $\omega$에서 $p_1^\omega$의 support는 $S_1$, 실현 가능 조각 집합은 $F_1$이고, $(p_0, p_1^\omega) \in \mathcal M_k(r)$이다.

(ii) $E_j = \{c_j \notin O,\ c_j' \notin O\}$라 하자. 임의의 oversampler $\hat q$에 대해
$$\max_{\omega} \mathbb E_\omega\, \mathcal A(\hat q) \ \ge\ \eta \sum_{j=1}^J m_j \ \min_{\omega} \mathbb E_\omega\!\left[\frac{\mathbf 1_{E_j}}{M_0}\right].$$
여기서 $\mathbb E_\omega$는 $(p_0, p_1^\omega)$ 아래 기댓값이고 $M_0 = p_1^\omega(\mathrm{SZ})$이다.

(iii) 모든 $\omega$, $j$에서
$$\mathbb E_\omega\!\left[\frac{\mathbf 1_{E_j}}{M_0}\right] \ \ge\ \frac{(1 - a_j - b_j)^{n_1}}{a_j + b_j + \mathbb E_\omega M_0}.$$

**따름정리 A.** 모든 $j$에서 $n_1(a_j + b_j) \le \varepsilon$이고, $\bar M = \max_\omega \mathbb E_\omega M_0$, $\sum_j m_j \ge \theta \bar M$이면
$$\max_{\omega} \mathbb E_\omega\, \mathcal A(\hat q) \ \ge\ \eta\,\theta\,(1 - \varepsilon)\, \frac{\bar M}{\bar M + \varepsilon / n_1}.$$
$\bar M \gg 1/n_1$이면 우변은 $\eta\theta(1-\varepsilon)$에 가깝다. $\theta$는 짝지을 수 있는 드문 셀이 기대 missing mass에서 차지하는 몫이다($\sum_j m_j \le \tfrac12 \sum_j (a_j + b_j)$이므로 $\theta$는 대략 $1/2$ 이하이다). 따라서 "균일하게 작을 수 없다"는 결론은 $\theta$가 0에서 떨어져 있을 때, 즉 $n_1(a_j + b_j)$가 작은 드문 셀이 missing mass의 상당 부분을 차지하는 영역에서만 성립한다. $n_1$이 support에 비해 커서 missing mass가 드문 셀에 거의 없으면 $\theta \to 0$이고 하한도 0으로 간다.

**조건 (S)의 충분조건 (S′).** $(p_0, p_1) \in \mathcal M_k(r')$이고 $r' > r$, $\eta \le 1 - r/r'$이면 (S)가 성립한다.

### 2.3 증명

**(S′) ⇒ (S).** $j \in J(\beta)$이면 쌍 $j$의 셀 중 정확히 하나가 $\beta$에 있고 그 질량은 $m_j$ 이상이다. 서로 다른 $j$는 서로 다른 셀을 주므로 $\sum_{j \in J(\beta)} m_j \le \pi_1(\beta)$이다. 따라서 $\pi_1(\beta)(1 - \eta) \ge r\pi_0(\beta)$이면 (S)가 성립한다. $\pi_1(\beta) \ge r'\pi_0(\beta)$이고 $(1-\eta) r' \ge r$이면 이 부등식이 성립한다.

**(i).** $p_1^\omega(c_j) \ge a_j - \eta m_j \ge (1-\eta) a_j > 0$이고 $c_j'$도 같다. 쌍 밖의 셀은 질량이 그대로이므로 $p_1^\omega$의 support는 $S_1$이다. 조각 $\beta$의 실현 가능 여부는 $\pi_1^\omega(\beta) = p_1^\omega(\beta \cap S_1) > 0$인지로 정해지므로 support에만 의존한다. 따라서 실현 가능 조각 집합도 $F_1$이고, 가정 1은 $(p_0, p_1)$에서와 같은 문장이 되어 성립한다. 가정 2에 대해서는
$$\pi_1^\omega(\beta) - \pi_1(\beta) = \sum_j s_j \eta m_j \big(\mathbf 1\{c_j \in \beta\} - \mathbf 1\{c_j' \in \beta\}\big) \ \ge\ -\eta \sum_{j \in J(\beta)} m_j$$
이므로 (S)에서 $\pi_1^\omega(\beta) \ge r\pi_0(\beta)$가 모든 $\beta \in F_1$에서 성립한다.

**(ii).** $\omega^{(j)}$는 $\omega$에서 $j$번째 좌표만 바꾼 것이다.

*1단계 (쌍별 손실).* $\hat q(\mathrm{SZ}) > 0$이면 $Q = \hat q(\cdot \mid \mathrm{SZ})$, $\hat q(\mathrm{SZ}) = 0$이면 $Q \equiv 0$으로 두고
$$\ell_j(\omega) = \mathbf 1_{E_j} \cdot \tfrac12 \sum_{c \in \{c_j, c_j'\}} \big|Q(c) - P^\star_\omega(c)\big|, \qquad P^\star_\omega = p_1^\omega(\cdot \mid \mathrm{SZ})$$
라 하자. $E_j$ 위에서는 $c_j, c_j' \in \mathrm{SZ}$이고 쌍들이 서로 겹치지 않는다. 그래서 $\hat q(\mathrm{SZ}) > 0$이면 $\mathcal A(\hat q) \ge \sum_j \ell_j(\omega)$이다. $\hat q(\mathrm{SZ}) = 0$이면 $\sum_j \ell_j(\omega) \le \tfrac12 \sum_c P^\star_\omega(c) \le \tfrac12 < 1 = \mathcal A$이다. $M_0 = 0$이면 $E_j$가 일어날 수 없으므로($E_j$ 위에서 $M_0 \ge a_j + b_j > 0$) 모든 $\ell_j = 0$이다. 따라서 어느 경우든 $\mathcal A(\hat q) \ge \sum_j \ell_j(\omega)$이다.

*2단계 ($E_j$ 위에서 두 법칙이 같음).* $(D_0, D_1, \xi)$의 법칙을 $L_\omega$라 하자. $D_0$와 $\xi$의 법칙은 $\omega$와 무관하다. $D_1$의 확률은 $\prod_{i} p_1^\omega(C_{1,i})$이다. $\omega$와 $\omega^{(j)}$는 셀 $c_j, c_j'$에서만 다르고 $E_j$ 위에서는 어느 행도 이 두 셀이 아니다. 그래서 $E_j$ 위에서 두 확률이 점별로 같다. 따라서 $E_j$ 밖에서 0인 데이터의 함수 $h \ge 0$에 대해 $\mathbb E_{\omega^{(j)}} h = \mathbb E_\omega h$이다.

*3단계 (분리).* $E_j$ 위에서 $M_0$는 $\omega$와 $\omega^{(j)}$에서 같다. 두 셀이 모두 SZ에 있고 쌍의 질량 합 $a_j + b_j$가 불변이며, 나머지 셀의 질량과 $O$가 같기 때문이다. 그래서 $E_j$ 위에서
$$|P^\star_\omega(c) - P^\star_{\omega^{(j)}}(c)| = \frac{2\eta m_j}{M_0}, \qquad c \in \{c_j, c_j'\}.$$
$Q$는 데이터만의 함수이므로 삼각부등식에서
$$\ell_j(\omega) + \ell_j(\omega^{(j)}) \ \ge\ \mathbf 1_{E_j} \cdot \tfrac12 \sum_{c \in \{c_j, c_j'\}} |P^\star_\omega(c) - P^\star_{\omega^{(j)}}(c)| = \frac{2\eta m_j}{M_0}\mathbf 1_{E_j}.$$
$\ell_j(\omega^{(j)})$는 $E_j$ 밖에서 0이므로 2단계에서 $\mathbb E_{\omega^{(j)}} \ell_j(\omega^{(j)}) = \mathbb E_\omega \ell_j(\omega^{(j)})$이고, 따라서
$$\mathbb E_\omega \ell_j(\omega) + \mathbb E_{\omega^{(j)}} \ell_j(\omega^{(j)}) \ \ge\ 2\eta m_j\, \mathbb E_\omega\!\left[\frac{\mathbf 1_{E_j}}{M_0}\right].$$

*4단계 (Assouad).*
$$\max_\omega \mathbb E_\omega \mathcal A \ \ge\ 2^{-J}\sum_\omega \mathbb E_\omega \mathcal A \ \ge\ \sum_j 2^{-J} \sum_{\omega:\,\omega_j = 0} \big[\mathbb E_\omega \ell_j(\omega) + \mathbb E_{\omega^{(j)}} \ell_j(\omega^{(j)})\big] \ \ge\ \sum_j \tfrac12 \cdot 2\eta m_j \min_\omega \mathbb E_\omega\!\left[\frac{\mathbf 1_{E_j}}{M_0}\right].$$

**(iii).** 쌍 $j$의 질량 합은 $a_j + b_j$로 $\omega$와 무관하므로 $P_\omega(E_j) = (1 - a_j - b_j)^{n_1}$이다. $E_j$ 위에서 $M_0 \ge a_j + b_j > 0$이므로 Jensen 부등식에서 $\mathbb E[1/M_0 \mid E_j] \ge 1/\mathbb E[M_0 \mid E_j]$이다. $E_j$가 주어지면 $D_1$의 행은 $p_1^\omega(\cdot \mid U \setminus \{c_j, c_j'\})$에서 독립으로 뽑힌다. 그래서 쌍 밖의 셀 $c$에 대해
$$P_\omega(c \notin O \mid E_j) = \Big(1 - \frac{p_1^\omega(c)}{1 - a_j - b_j}\Big)^{n_1} \le \big(1 - p_1^\omega(c)\big)^{n_1} = P_\omega(c \notin O)$$
이다. 따라서 $\mathbb E_\omega[M_0 \mid E_j] \le a_j + b_j + \mathbb E_\omega M_0$이다.

**따름정리 A.** (iii)에 Bernoulli 부등식 $(1 - a_j - b_j)^{n_1} \ge 1 - n_1(a_j + b_j) \ge 1 - \varepsilon$과 $a_j + b_j \le \varepsilon/n_1$을 넣고 (ii)에 대입한다. $\blacksquare$

### 2.4 예: 균등 법칙

$p_0 = p_1 =$ $U$ 위의 균등분포, $|U| = N$(짝수), $S_1 = U$로 두고 $U$를 $N/2$개 쌍으로 나눈다. 모든 조각이 실현 가능하므로 가정 1이 성립하고, $r' = 1$이므로 (S′)에서 $\eta \le 1 - r$이면 된다. $a_j = b_j = m_j = 1/N$, $\sum_j m_j = 1/2$, $\mathbb E M_0 \le 1$이므로
$$\max_\omega \mathbb E_\omega \mathcal A(\hat q) \ \ge\ \frac{\eta}{2} \cdot \frac{(1 - 2/N)^{n_1}}{1 + 2/N}.$$
$r = 1/2$, $\eta = 1/2$, $n_1 \ll N$이면 약 $1/4$이다. SZ 위의 균등분포를 내는 절차는 각 $\omega$에서 $\mathrm{TV} \approx \eta/2$를 얻으므로, 이 족에서 하한은 $1 - O(n_1/N)$ 배 차이로 달성된다.

### 2.5 주석

1. **majority 표본과 무관하다.** 족 전체에서 $p_0$가 같으므로 $n_0$가 아무리 커도, $p_0$를 정확히 알아도 하한은 그대로이다.
2. **support를 알아도 마찬가지이다.** 족 전체에서 $S_1$이 같으므로, $\hat S = S_1$을 아는 절차도 하한을 피하지 못한다. 정리 3의 사건 $\{\hat S = S_1\}$은 배분을 정하지 못한다.
3. **정리 3과 충돌하지 않는다.** 족의 모든 법칙이 $\mathcal M_k(r)$에 있으므로, 조건 (7)이 성립하면 정리 3의 총질량 보장은 각 $\omega$에서 그대로 성립한다. 막히는 것은 총질량을 셀 사이에 나누는 방법뿐이다.
4. **말하지 않는 것.** 초안 Lemma 4.1은 거리별 몫의 분자와 분모에 대한 (크기 $n_1 - 1$에서의) 불편 추정량을 준다. 초안은 그 비율 $\hat\pi_s$가 불편이거나 일치 추정량이라고 주장하지 않는다(495–496줄). 하한은 이런 여러 셀에 걸친 양과 모순되지 않는다. 하한이 막는 것은 셀 단위 배분을 $\mathcal M_k(r)$ 전체에서 균일하게 맞추는 것이다. 특정 법칙에서는 어떤 방법이 잘 맞출 수 있다(예: 디코더의 귀납 편향이 그 법칙과 맞을 때).
5. **제곱 오차 형태.** $L_2(\hat q) = \sum_{c \in \mathrm{SZ}} (Q(c) - P^\star(c))^2$($\hat q(\mathrm{SZ}) = 0$이면 1단계와 같이 $Q \equiv 0$)에 대해서도, $(x - u)^2 + (x - v)^2 \ge (u - v)^2/2$를 쓰면 같은 논증으로
$$\max_\omega \mathbb E_\omega L_2(\hat q) \ \ge\ 2\eta^2 \sum_j m_j^2 \min_\omega \mathbb E_\omega\!\left[\frac{\mathbf 1_{E_j}}{M_0^2}\right]$$
이다. 2.4절 예에서는 약 $\eta^2/N$이고, $\|P^\star\|_2^2 \approx 1/N$ 대비 비율이 $\eta^2$이다. 셀별 오차의 제곱합으로 재도 결론은 같다.
6. **상수는 최적화하지 않았다.** 쌍 대신 더 큰 묶음을 쓰면 상수를 키울 수 있으나 결론에는 필요하지 않다.

## 3. 명제 B (배분 오차 분해)

### 3.1 설정

초안의 법칙 (4)를 쓴다.
$$\hat q = (1 - w) Q_O + w Q_N, \qquad w = G\,\mathbf 1\{\hat N \ne \emptyset\}.$$
- $Q_O$는 $O$ 위의 법칙이다.
- $\hat N \subseteq \hat S \setminus O$는 novel stratum이다. 초안 3.4절 규칙에 따라 $O \cap O_0 = \emptyset$이고 $\hat S \setminus (O \cup O_0) \ne \emptyset$이면 $\hat N = \hat S \setminus (O \cup O_0)$, 그 밖에는 $\hat N = \hat S \setminus O$이다.
- **실효 후보 집합.** 구현은 셀의 $O$까지 Hamming 거리마다 후보를 나열하거나(listed distance) 디코더에서 뽑는다(drawn distance). 첫 디코딩 배치에 $O \cup O_0$ 밖의 행이 없으면, drawn distance는 다음 배치부터 $\hat S \setminus O$를 쓴다. listed distance는 계속 $O_0$를 제외한다(초안 부록 G, 1810–1812줄). 그래서 선택된 셀이 공식 $\hat N$ 밖에 있을 수 있다. 실제로 제외가 유지된 셀의 집합을
$$X = \begin{cases} O_0 \cap (\hat S \setminus O) \cap \{c : c\text{의 거리가 listed이거나, drawn이고 전환이 일어나지 않음}\} & \text{제외 규칙이 적용될 때} \\ \emptyset & \text{그 밖} \end{cases}$$
로 두고, 실효 후보 집합을 $\hat N_{\text{eff}} = (\hat S \setminus O) \setminus X$로 정의한다. 할당이 0인 거리의 셀도 $\hat N_{\text{eff}}$에 넣는다. 그래서 거리별 할당의 효과는 $T_1$이 아니라 $T_{3a}$에 들어간다(초안 474–478줄의 예가 이 경우이다). 이하 $\hat N$은 $\hat N_{\text{eff}}$를 뜻한다.
- $Q_N$은 $\hat N$ 위의 법칙이고 그 support를 $\hat K$(선택된 셀)라 한다. BOMiS에서 $Q_N$은 선택된 셀의 같은 수 사본으로 이루어진 $n_{\text{new}}$개 행의 경험 법칙이다. posterior filter, Chao1 개수, 거리별 할당, 개수 보충은 모두 $\hat K$를 정하는 단계이다. 위 정의에서 $\hat K \subseteq \hat N$이다. 선택된 셀이 없으면(shortfall) 초안은 $\hat q = Q_O$로 둔다(371–373줄).
- 디코더의 novel 법칙은 초안 3.4절의 $Q$를 $\hat N$ 위로 옮긴 $Q(c) = \tilde\pi(c)\mathbf 1\{c \in \hat N\}/\tilde\pi(\hat N)$이다. 초안은 $\tilde\pi > 0$이 $\hat S$ 위에서 성립한다고 쓴다(255–256줄). 이에 따라 $\hat N \ne \emptyset$이면 $Q$는 잘 정의되고 $\hat N$의 모든 셀에서 양수이다. 원저자 코드에서도 이 성질을 확인했다(4.1절).

사건 $H = \{\hat S \subseteq S_1,\ \hat K \ne \emptyset,\ G > 0\}$. $\hat K \ne \emptyset$이면 $\hat N \ne \emptyset$이다.

### 3.2 명제

**명제 B.** $H$ 위에서 $\mathcal A(\hat q) = \mathrm{TV}(Q_N, P^\star)$이고
$$1 - \frac{p_1(\hat K)}{M_0} \ \le\ \mathcal A(\hat q) \ \le\ T_1 + T_2 + T_3,$$
$$T_1 = 1 - \frac{p_1(\hat N)}{M_0}, \qquad T_2 = \mathrm{TV}\big(Q,\ p_1(\cdot \mid \hat N)\big), \qquad T_3 = \mathrm{TV}(Q_N, Q).$$
더 나누면
$$T_1 = \underbrace{\frac{p_1(\mathrm{SZ} \setminus \hat S)}{M_0}}_{T_{1a}} + \underbrace{\frac{p_1\big((\hat S \setminus O) \setminus \hat N\big)}{M_0}}_{T_{1b}}, \qquad \underbrace{Q(\hat N \setminus \hat K)}_{T_{3a}} \ \le\ T_3 \ \le\ T_{3a} + \underbrace{\mathrm{TV}\big(Q_N,\ Q(\cdot \mid \hat K)\big)}_{T_{3b}}.$$

| 항 | 뜻 | 원인 |
|---|---|---|
| $T_{1a}$ | $\hat S$ 밖으로 빠진 SZ의 질량 몫 | support 추정(인증 오류, 허용 안 된 조각) |
| $T_{1b}$ | $O_0$ 제외로 실제로 빠진 SZ의 질량 몫 $p_1(\mathrm{SZ} \cap X)/M_0$. 제외 규칙이 적용되지 않으면 0 | $O_0$ 제외 규칙 |
| $T_2$ | 디코더가 $\hat N$ 위에서 $p_1$의 모양을 틀린 정도 | 디코더 |
| $T_{3a}$ | 선택되지 않은 셀에 디코더가 준 질량 | filter, Chao1 개수, 거리별 할당 |
| $T_{3b}$ | 선택된 셀 안에서 같은 수 사본과 디코더 비율의 차이 | 같은 수 사본 |

하한 $1 - p_1(\hat K)/M_0$는 초안 5절의 coverage를 쓰면 $1 - \text{coverage}$이다. 배분 오차는 덮지 못한 SZ 질량 몫보다 작을 수 없다.

### 3.3 증명

**보조 사실.** 법칙 $P$와 $P(B) > 0$인 집합 $B$에 대해 $\mathrm{TV}(P(\cdot \mid B), P) = 1 - P(B)$이다. $c \in B$에서 $P(c \mid B) - P(c) = P(c)(1 - P(B))/P(B) \ge 0$이고 그 합은 $1 - P(B)$이다. $c \notin B$에서는 차이가 $P(c)$이고 그 합도 $1 - P(B)$이다. 두 합의 절반이 $1 - P(B)$이다. 같은 방법으로, $Q'(B^c) = 0$인 임의의 법칙 $Q'$에 대해 $\mathrm{TV}(Q', P) \ge P(B^c)$이다(집합 $B^c$에서의 차이).

**$\mathcal A = \mathrm{TV}(Q_N, P^\star)$.** $H$ 위에서 $\hat S \subseteq S_1$이므로 $\hat N \subseteq \hat S \setminus O \subseteq \mathrm{SZ}$이다. $\hat K \ne \emptyset$이므로 shortfall이 아니고 $\hat q$는 법칙 (4)의 형태이다. $\hat N \supseteq \hat K \ne \emptyset$이고 $p_1 > 0$이 $S_1$ 위에서 성립하므로 $M_0 \ge p_1(\hat N) > 0$이다. $w = G > 0$이다. $Q_O$는 $O$ 위에 있어 SZ와 겹치지 않고, $Q_N$은 $\hat N \subseteq \mathrm{SZ}$ 위에 있다. 따라서 $\hat q$를 SZ로 제한하면 $wQ_N$이고, $\hat q(\mathrm{SZ}) = w > 0$, $\hat q(\cdot \mid \mathrm{SZ}) = Q_N$이다.

**상한.** 삼각부등식으로
$$\mathrm{TV}(Q_N, P^\star) \le \mathrm{TV}(Q_N, Q) + \mathrm{TV}\big(Q, P^\star(\cdot \mid \hat N)\big) + \mathrm{TV}\big(P^\star(\cdot \mid \hat N), P^\star\big).$$
$\hat N \subseteq \mathrm{SZ}$이므로 $P^\star(\cdot \mid \hat N) = p_1(\cdot \mid \hat N)$이고, 보조 사실에서 마지막 항은 $1 - P^\star(\hat N) = 1 - p_1(\hat N)/M_0 = T_1$이다.

**$T_1$ 분해.** $\mathrm{SZ} \setminus \hat N$은 $\mathrm{SZ} \setminus \hat S$와 $(\mathrm{SZ} \cap \hat S) \setminus \hat N$의 서로소 합집합이다. $\hat S \subseteq S_1$이면 $\mathrm{SZ} \cap \hat S = \hat S \setminus O$이고, $\hat N = \hat N_{\text{eff}}$의 정의에서 $(\hat S \setminus O) \setminus \hat N = X$이다.

**$T_3$ 분해.** 삼각부등식과 보조 사실로 $\mathrm{TV}(Q_N, Q) \le \mathrm{TV}(Q_N, Q(\cdot \mid \hat K)) + \mathrm{TV}(Q(\cdot \mid \hat K), Q) = T_{3b} + Q(\hat N \setminus \hat K)$이다. $Q(\hat K) > 0$은 $\hat K \subseteq \hat N$이 공집합이 아니고 $Q > 0$이 $\hat N$ 위에서 성립하므로 맞다. $Q_N(\hat N \setminus \hat K) = 0$이므로 보조 사실의 두 번째 부분에서 $T_3 \ge T_{3a}$이다.

**하한.** $Q_N(\mathrm{SZ} \setminus \hat K) = 0$이므로 보조 사실의 두 번째 부분에서 $\mathrm{TV}(Q_N, P^\star) \ge P^\star(\mathrm{SZ} \setminus \hat K) = 1 - p_1(\hat K)/M_0$이다. $\blacksquare$

### 3.4 행 수준과 기댓값 형태

**행 수준.** 알고리즘 1의 출력에서 novel 행은 $\hat K$의 같은 수 사본이다. 부록 G의 거리별 상한 $m_s = \min\{q_s, \max(1, \lceil \text{ncell}\,\hat\pi_s \rceil)\}$에서 $q_s$는 ncell의 최대잉여 할당이므로, 선택된 셀 수는 ncell 이하이고 $\text{ncell} \le n_{\text{new}}$이다. 그래서 선택된 셀은 모두 한 번 이상 나오고, SZ에 놓인 출력 행의 경험 법칙은 $Q_N$과 같다. $H$의 $G > 0$을 $n_{\text{new}} \ge 1$로 바꾸면 명제 B는 출력 행에 그대로 성립한다. 이때 $1 - p_1(\hat K)/M_0$는 초안 5절에서 행으로 계산하는 coverage(669줄)와 같고, 원저자 코드 `metrics.evaluate`의 `recovery`와도 같다(4.1절).

**행 수준 항등식.** $H$ 위에서 $Q_N$의 support가 $\hat K \subseteq \mathrm{SZ}$이면
$$\mathcal A(\hat q) = \Big(1 - \frac{p_1(\hat K)}{M_0}\Big) + \sum_{c \in \hat K} \big(P^\star(c) - Q_N(c)\big)_+$$
이다. 증명: $a(c) = Q_N(c) - P^\star(c)$라 하면 $\sum_{c \in \hat K} a(c) = 1 - P^\star(\hat K)$이므로 $\sum_{\hat K} a_+ = (1 - P^\star(\hat K)) + \sum_{\hat K} a_-$이다. 따라서 $\mathrm{TV}(Q_N, P^\star) = \tfrac12\big[\sum_{\hat K}|a| + P^\star(\mathrm{SZ} \setminus \hat K)\big] = (1 - P^\star(\hat K)) + \sum_{\hat K} a_-$이다. 첫 항은 덮지 못한 SZ 질량(1 − coverage)이고, 둘째 항은 선택된 셀 안에서 참 몫보다 적게 받은 양의 합이다. 그래서 행 수준 $\mathcal A$가 기존 `recovery`에 더하는 정보는 둘째 항뿐이다.

**기댓값 형태.**
$$\mathbb E\,\mathcal A(\hat q) \le \mathbb E\big[(T_1 + T_2 + T_3)\mathbf 1_H\big] + P(H^c,\ M_0 > 0).$$
$\hat S \subseteq S_1$, $\hat K = \emptyset$, $M_0 > 0$인 경우는 둘로 나뉜다. $\hat S \setminus O = \emptyset$이면 $\mathrm{SZ} \ne \emptyset$이므로 $S_1 \not\subseteq \hat S$이다. $\hat S \setminus O \ne \emptyset$이면 탐색 shortfall이다. 따라서
$$\{H^c,\ M_0 > 0\} \subseteq \{\hat S \not\subseteq S_1\} \cup \{S_1 \not\subseteq \hat S\} \cup \{\hat S \setminus O \ne \emptyset,\ \hat K = \emptyset\} \cup \{G = 0,\ M_0 > 0\}.$$
- 첫 사건: 가정 1과 (7) 아래 확률 $\delta$ 이하(정리 2).
- 둘째 사건: 가정 2 아래 확률 $V + \alpha$ 이하(정리 3의 증명, 1243줄).
- 셋째 사건(탐색 shortfall): 초안도 가정으로 막지 못한다고 쓴다(472–473줄). 우변에 그대로 남긴다.
- 넷째 사건: 정리 3 증명의 Good-Turing 집중 부등식은 $\{G = 0,\ M_0 > a\}$만 막는다. $P(G = 0,\ 0 < M_0 \le a)$는 작다는 보장이 없고 그 위에서 $\mathcal A = 1$이다. 이 항도 우변에 그대로 남긴다.

분해 자체(점별 부등식)는 어떤 가정도 쓰지 않는다.

### 3.5 분해의 해석과 명제 A와의 연결

**해석.** $\mathcal A_Q = \mathrm{TV}(Q, P^\star)$를 디코더의 novel 법칙을 그대로 쓸 때의 배분 오차라 하자. $H$ 위에서 삼각부등식과 역삼각부등식으로
$$\mathcal A_Q \le T_1 + T_2, \qquad |\mathcal A - \mathcal A_Q| \le T_3$$
이다. $\mathcal A - \mathcal A_Q$는 선택 단계(filter, Chao1 개수, 거리별 할당, 기울이기, 같은 수 사본)의 순효과이고 음수일 수 있다. 선택 단계는 원래 $Q$의 모양을 단일 사본 몫 추정으로 고치려는 장치이기 때문이다(초안 3.5절). 따라서 $T_3$는 상한의 한 항일 뿐 순수한 비용이 아니다. $T_3$가 커도 $\mathcal A$는 작을 수 있다. 실험에서는 $\mathcal A$와 $\mathcal A_Q$를 함께 보고하고, $\mathcal A - \mathcal A_Q$의 부호로 선택 단계가 배분을 개선하는지 판단한다.

**명제 A와의 연결.** 명제 A는 BOMiS를 포함한 모든 절차의 $\mathcal A$에 직접 적용된다. 디코더 항에 대해서는 다음이 성립한다. $O_0$를 제외하지 않고($X = \emptyset$) $Q_N = Q$로 두는 변형에서는 $H$ 위에서 $\mathcal A = \mathcal A_Q \le T_{1a} + T_2$이므로, 명제 A의 족에서
$$\max_\omega \mathbb E_\omega[T_2 \mathbf 1_H] \ \ge\ \text{(명제 A의 하한)} - \max_\omega \Big(\mathbb E_\omega[T_{1a}\mathbf 1_H] + P_\omega(H^c, M_0 > 0)\Big)$$
이다. support 단계가 정확하고 shortfall이 드물어 오른쪽 괄호가 작으면, 디코더 항 $T_2$는 $\mathcal M_k(r)$ 전체에서 균일하게 작을 수 없다. 이 연결은 법칙 수준($Q$)의 진술이다. $Q_N = Q$를 유한 개 행으로 뽑으면 행 수준에서는 표집 오차가 더해진다.

## 4. 합성 실험에서의 계산

### 4.1 원저자 코드 확인 (github.com/suman-cha/bomis, commit fa088a9, `bomis/model.py`, `bomis/metrics.py`)

- **$Q > 0$ ($\hat S$ 위).** `decode`(319–341줄)는 `log_softmax` 뒤 허용 수준만 남기고 `exp(lp - max)`로 다시 정규화해 뽑는다. top-k, temperature, nucleus 절단은 없다. `log_prob`(343–362줄)은 마스크를 건 조건부를 수준마다 정규화해 더하므로 $\hat S$의 모든 셀에서 유한하고 $\hat S$ 밖에서 $-\infty$이다. 경로 확률이 양수인 셀은 `Support.admissible`이 마지막 좌표에서 모든 조각을 검사하므로 정확히 $\hat S$이다. 따라서 $Q(c) = \exp(\texttt{log\_prob}(c)) / \sum_{c' \in \hat N}\exp(\texttt{log\_prob}(c'))$로 정확히 계산할 수 있다. `decode`는 float32로 계산하므로 최대 수준 대비 확률이 약 $10^{-45}$ 미만인 수준은 실제로 뽑히지 않는다. 다만 $Q$는 `log_prob`로 계산하므로 영향이 없다.
- **listed/drawn 여부와 전환 여부는 기록되지 않는다.** `_novel`(428–541줄)에서 listed distance 집합 `exact`, 전환 여부 `excl_fallback`, 후보 `cand`·`pool`은 지역 변수이고 반환값은 행 `R`뿐이다. `sample`은 관측 행과 섞어서 반환한다. 다만 출력을 바꾸지 않고 복원할 수 있다. 아래 감싸기는 난수를 소비하지 않으므로 출력이 비트 단위로 같다.
  - `_novel(n_new, budget)` 감싸기: $n_{\text{new}}$와 반환 행 `R`을 기록한다. $\hat K$ = `R`의 서로 다른 행, shortfall = (`len(R) == 0`이고 $n_{\text{new}} \ge 1$).
  - `_shell_cells(s, limit)` 감싸기: 반환값이 `None`이 아니면 거리 $s$는 listed이다(`exact.add(s)`와 같은 조건, 455–459줄).
  - `decode(n, rng)` 감싸기: 첫 호출의 출력을 기록한다. 제외 규칙이 적용되고(`model.majority`가 비어 있지 않음) 첫 배치에 $O \cup O_0$ 밖의 행이 없으면 전환이 일어난 것이다(479–481줄).
  - 제외 규칙 적용 여부: 학습 후 `model.majority`가 비어 있지 않은지(212줄).
  - 셀의 거리: `model._dist(Z)`. 할당이 0인 거리는 listed 판정 대상이 아니므로(454줄 `for s in cap`) drawn 규칙(전환이 없으면 제외 유지)을 따른다고 둔다. 이 셀들은 선택될 수 없으므로 이 관례는 $T_{1b}$와 $T_{3a}$ 사이의 배분만 바꾼다.
- **선택 셀은 실효 후보 집합 안에 있다.** listed 후보는 `_shell_cells`에서 `support.contains`와 `self.majority` 제외를 거친다(410–412줄). drawn 후보는 `decode` 출력에서 `skip`(전환 전 $O \cup O_0$, 전환 후 $O$)을 뺀 것이다(478줄). 할당이 0인 거리의 뽑힌 셀은 `got`과 `pool`에 들어가지 않는다(495줄). 따라서 $\hat K \subseteq \hat N_{\text{eff}}$이다.
- **선택 셀 수.** $L = \min(n_{\text{new}}, \max(1, \lceil \hat f_0 \rceil))$이고 거리별 상한은 $\min(q_s, \max(1, \lceil L\,\hat\pi_s \rceil))$이다(446–448줄). 그래서 $|\hat K|$는 대략 $\lceil \hat f_0 \rceil$에 거리 개수를 더한 값 이하이다. 행은 `R[np.arange(n_new) % len(R)]`로 같은 수 사본이 된다(541줄).
- **`recovery` = coverage.** `metrics.evaluate`의 `recovery`는 생성 행 중 $S_1 \setminus O$에 든 서로 다른 셀의 $p_1$ 합을 $M_0$로 나눈 값이다(53–57줄). 초안의 recovered mass $\Gamma$(행 비율)가 아니라 coverage이다. ADEMP 7.1절의 `recovery`도 이 정의이다.

### 4.2 행 수준 $\mathcal A$가 방법을 구분하는가

3.4절 항등식에 따라 행 수준 $\mathcal A = (1 - \texttt{recovery}) + \sum_{\hat K}(P^\star - Q_N)_+$이다(누출이 없을 때. 누출이 있으면 $\hat K \cap S_1$로 읽는다). 그래서 다음 두 가지가 정해진다.

- `recovery`가 모든 방법에서 0 근처이면 $\mathcal A$는 모든 방법에서 1 근처가 되어 방법을 구분하지 못한다.
- `recovery`가 0에서 떨어져 있으면 $\mathcal A$는 `recovery`에 "선택된 셀 안에서 덜 받은 양"을 더한 지표가 된다.

초안 Table 1(개발 seed 0)의 HM과 $\hat\Lambda$에서 BOMiS의 행 수준 coverage를 거꾸로 계산하면, enumerated domain에서 약 0.50, sampled table에서 약 0.29이다($\text{cov} = \text{HM}\cdot p/(2p - \text{HM})$, $p = 1 - \hat\Lambda$). 모집단 평균의 HM에서 계산했으므로 대략값이다. 같은 방식으로 ARF는 약 0.31과 0.20, TabDiff는 enumerated에서 약 0.32이다. 초안의 모집단에서는 coverage가 0 근처가 아니므로 $\mathcal A$가 1에 붙지 않고, 방법마다 하한도 다르다. splice는 $M_0 = 1$이라 모든 방법의 coverage가 0 근처이고(초안 683–684줄), 이 경우에는 구분하지 못한다.

우리 합성 DGP에서의 값은 코드만으로는 정해지지 않는다. 다만 2단계·4단계 개발 실행이 `recovery`를 이미 계산하므로, 그 값을 보면 판단된다. 별도 실행은 필요 없다.

### 4.3 계산 항목

- $p_1$을 정확히 알므로 SZ, $M_0$, $P^\star$가 정확하다.
- $\hat S$, $\hat N$, $\hat K$, $Q_N$은 실행 기록에서 얻는다.
- $\hat S$, $\hat K$, $Q_N$은 실행 기록에서 얻는다. $\hat N = \hat N_{\text{eff}}$를 만들려면 거리별 listed/drawn 여부와 전환 여부($X$)가 필요하고, 4.1절의 감싸기로 얻는다.
- $Q$에는 $\hat N$ 위의 $\tilde\pi(c)$가 필요하다. $U$가 $4^8 = 65{,}536$개 셀이므로 마스크를 건 디코더의 `log_prob`를 모든 셀에서 계산할 수 있다. 모든 항이 Monte Carlo 없이 정확값이다.
- 기록할 양: $\mathcal A$(ADEMP 7.2절 (b)의 TV), $\mathcal A_Q$, $T_{1a}, T_{1b}, T_2, T_{3a}, T_{3b}$, 행 수준 coverage. $T_{1b}$는 1번 SZ($\mathrm{SZ} \cap O_0$) 중 실제로 제외된 것의 질량 몫이다.
- 행 수준 $\mathcal A$의 변별력은 4.2절. 개발 실행의 `recovery`가 0 근처이면 법칙 수준 $\mathcal A_Q$를 주 지표로 본다.

## 5. 남은 확인 사항

- (해결) $\tilde\pi$의 양수성과 기록 여부는 4.1절에서 원저자 코드로 확인했다. 기록용 감싸기 세 개는 구현이 필요하다(Codex 경유, 출력이 비트 단위로 같은지 확인).
- 명제 A는 기댓값 하한이다. 확률 형태가 필요하면 같은 논증을 사건에 대해 다시 쓴다.
- 명제 A를 본문 명제로 넣을지, remark나 부록으로 넣을지는 연구자가 정한다.
