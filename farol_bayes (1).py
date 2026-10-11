"""
Trabalho 2 - Inteligência Artificial (UFAM/IComp)
Rede bayesiana para diagnóstico de farol de bicicleta a dínamo.

Python puro (sem dependências). Implementa inferência por enumeração
sobre a distribuição conjunta fatorada pela rede:

P(Str,Flw,R,V,B,K,Li) = P(Str) P(Flw) P(B) P(K) P(R|Str,Flw) P(V|R) P(Li|V,B,K)

Execute:  python3 farol_bayes.py
"""
from itertools import product

# ---------------- CPTs (valores plausíveis - item (c)) ----------------
STR = ["dry", "wet", "snow_covered"]
P_STR = {"dry": 0.65, "wet": 0.28, "snow_covered": 0.07}
P_FLW = {True: 0.075, False: 0.925}
P_B = {True: 0.95, False: 0.05}
P_K = {True: 0.98, False: 0.02}

# P(R=t | Str, Flw)
P_R_T = {
    ("dry", True): 0.40, ("dry", False): 0.05,
    ("wet", True): 0.70, ("wet", False): 0.20,
    ("snow_covered", True): 0.90, ("snow_covered", False): 0.40,
}
# P(V=t | R)
P_V_T = {True: 0.10, False: 0.95}
# P(Li=t | V,B,K)  -- tabela dada no enunciado
P_LI_T = {
    (True, True, True): 0.99, (True, True, False): 0.01,
    (True, False, True): 0.01, (True, False, False): 0.001,
    (False, True, True): 0.3, (False, True, False): 0.005,
    (False, False, True): 0.005, (False, False, False): 0.0,
}

def p_r(r, s, f):
    return P_R_T[(s, f)] if r else 1 - P_R_T[(s, f)]

def p_v(v, r):
    return P_V_T[r] if v else 1 - P_V_T[r]

def p_li(li, v, b, k):
    return P_LI_T[(v, b, k)] if li else 1 - P_LI_T[(v, b, k)]

def joint(s, f, r, v, b, k, li):
    return (P_STR[s] * P_FLW[f] * P_B[b] * P_K[k] *
            p_r(r, s, f) * p_v(v, r) * p_li(li, v, b, k))

B = (True, False)
WORLD = [dict(s=s, f=f, r=r, v=v, b=b, k=k, li=li)
         for s in STR for f in B for r in B for v in B
         for b in B for k in B for li in B]

def prob(**evid):
    """P(evidência) somando a conjunta sobre todas as variáveis ocultas."""
    return sum(joint(**w) for w in WORLD
               if all(w[x] == val for x, val in evid.items()))

def cond(query, given):
    return prob(**query, **given) / prob(**given)

if __name__ == "__main__":
    print("Soma da conjunta (deve ser 1):", round(prob(), 10))

    # ---- (e) P(V | Str = snow_covered)
    pv = cond(dict(v=True), dict(s="snow_covered"))
    print("\n(e) P(V=t | Str=snow_covered) =", round(pv, 4))
    print("    P(V=f | Str=snow_covered) =", round(1 - pv, 4))
    print("    P(R=t | Str=snow_covered) =",
          round(cond(dict(r=True), dict(s="snow_covered")), 4))
    print("    (comparação) P(V=t) sem evidência =", round(prob(v=True), 4))

    # ---- (d) ausência da aresta (Str, Li)
    print("\n(d) Str e Li são independentes dado V? P(Li=t | V,Str) para cada Str:")
    for v in B:
        vals = [round(cond(dict(li=True), dict(v=v, s=s)), 6) for s in STR]
        print(f"    V={v!s:5}:", dict(zip(STR, vals)))
    print("    (iguais entre as ruas => I(Str, Li | V))")
    print("    Marginalmente, P(Li=t | Str):",
          {s: round(cond(dict(li=True), dict(s=s)), 5) for s in STR},
          "(diferentes => dependência indireta via R e V)")

    # ---- independências do enunciado
    print("\nChecagem das hipóteses do enunciado:")
    print("    P(V=t|R=t,Str=snow) =", round(cond(dict(v=True), dict(r=True, s="snow_covered")), 4),
          "= P(V=t|R=t) =", round(cond(dict(v=True), dict(r=True)), 4))
    print("    P(Li=t|V=t,R=t) =", round(cond(dict(li=True), dict(v=True, r=True)), 5),
          "= P(Li=t|V=t) =", round(cond(dict(li=True), dict(v=True)), 5))
    print("    P(B=t,K=t) =", round(prob(b=True, k=True), 4),
          "= P(B=t)P(K=t) =", round(prob(b=True) * prob(k=True), 4))

    # ---- número de parâmetros
    rede = (len(STR) - 1) + 1 + 1 + 1 + 3 * 2 + 2 + 8
    cheia = 3 * 2 ** 6 - 1
    print(f"\nParâmetros independentes: rede = {rede}, conjunta completa = {cheia}")
