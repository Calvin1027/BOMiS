"""모집단 하나의 majority 구성을 단계별로 출력한다: 표 → 부모 쌍 0 → 심기 → p0 확인. 사용: show_tables.py <pop_seed>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from dgp.laws import apply_zeros, choose_parent_zeros, draw_tree_tables, tree_law  # noqa: E402
from dgp.minority import plant_pairs  # noqa: E402
from dgp.population import _rng  # noqa: E402
from dgp.space import CHAIN_EDGES, LEAF_EDGES, pattern_id, pattern_probs  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
name = {j: f"P{j + 1}" if j < 4 else f"L{j - 3}" for j in range(8)}

t = draw_tree_tables(_rng(seed, 0), 1.0)
print(f"== 모집단 seed {seed}: 3단계 표 (Dirichlet(1)) ==")
print("뿌리 P(P1):", t.root)
for j, l in CHAIN_EDGES + LEAF_EDGES:
    print(f"\nP({name[l]} | {name[j]})  행 = {name[j]} 값 0–3, 열 = {name[l]} 값 0–3")
    print(t.cond[j, l])

p_tree = tree_law(t)
pi_tree = pattern_probs(p_tree)
shared, major = choose_parent_zeros(p_tree, _rng(seed, 2))
print("\n== 4단계: 부모 쌍 0 (괄호 = 0으로 만들기 전 쌍 확률) ==")
for kind, zs in (("공유 0", shared), ("Z 쌍(majority 전용 0)", major)):
    for j, l, v, w in zs:
        print(f"{kind}: ({name[j]}={v}, {name[l]}={w})  ({pi_tree[pattern_id((j, l), (v, w))]:.4f})")

zeros = shared + major
pi_z = pattern_probs(apply_zeros(p_tree, zeros))
print("\n== 5단계 준비: 0 적용 뒤 부모 주변확률 ==")
for i in range(4):
    print(f"P0({name[i]}):", np.array([pi_z[pattern_id((i,), (v,))] for v in range(4)]))

t2, rules = plant_pairs(t, zeros, _rng(seed, 3))
print("\n== 5단계: 심기 (잎 표에서 바뀐 행) ==")
for (i, leaf), r_f in zip(LEAF_EDGES, rules["frequent"]):
    a = r_f.vals[0]
    print(f"\n{name[leaf]} | {name[i]}={a}  (a' = 가장 흔한 부모 값, P0 = {pi_z[pattern_id((i,), (a,))]:.3f})")
    print("  심기 전:", t.cond[i, leaf][a])
    print("  심은 뒤:", t2.cond[i, leaf][a])
    for lv in ("frequent", "mid", "rare"):
        r = [x for x in rules[lv] if x.cols == (i, leaf)][0]
        print(f"  {lv:8s} 금지 쌍 ({name[i]}={r.vals[0]}, {name[leaf]}={r.vals[1]}): e = {r.e:.3f}, 목표 pi0 = {r.pi0}")

p0 = apply_zeros(tree_law(t2), zeros)
pi0 = pattern_probs(p0)
print("\n== 6단계: 최종 p0에서 확인 ==")
for r in rules["frequent"]:
    print(f"pi0({name[r.cols[0]]}={r.vals[0]}, {name[r.cols[1]]}={r.vals[1]}) = {pi0[pattern_id(r.cols, r.vals)]:.6f}")
print("p0 > 0인 셀 수:", int((p0 > 0).sum()), "/ 65536")
