from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD

def read_instance(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    n, m = map(int, lines[0].strip().split())
    N = range(n)
    M = range(m)

    p = {}
    r = {}
    bigM = 0

    for i in N:
        for l in M:
            for k in M:
                r[i, l, k] = 0

    for i in N:
        line = list(map(int, lines[i + 1].strip().split()))
        for l in M:
            k = line[2 * l]
            t = line[2 * l + 1]

            p[i, k] = t
            r[i, l, k] = 1
            bigM += t

    return N, M, p, r, bigM


def manne_modeling(N, M, p, r, BigM):
    # Obtém os tamanhos de n e m
    n_len = len(N)
    m_len = len(M)

    # Cria o modelo
    model = LpProblem(name="Minimize_Makespan", sense=LpMinimize)

    # Variáveis de Decisão
    s = LpVariable.dicts(name='s',
                         indices=(N, M),
                         cat='Continuous',
                         lowBound=0)

    Z = LpVariable.dicts(name='Z',
                         indices=(N, N, M),
                         cat='Binary')

    # Cmax: makespan (tempo de conclusão da última tarefa)
    Cmax = LpVariable(name="Cmax", cat='Continuous', lowBound=0)

    # Função Objetivo
    model += Cmax, "Minimize_Makespan"

    # Restrições

    # 1. Restrições de Precedência (dentro de cada tarefa)
    for i in N:
        for l in range(m_len - 1):
            model += (
                    lpSum(r[i, l, k] * (s[i][k] + p[i, k]) for k in M)
                    <=
                    lpSum(r[i, l + 1, k] * s[i][k] for k in M)
            )

    # 2. Restrições de Disjunção (operações na mesma máquina)
    for k in M:
        for i in N:
            for u in N:
                if i < u:
                    if (i, k) in p and (u, k) in p:
                        model += (BigM + p[u, k]) * Z[i][u][k] + (s[i][k] - s[u][k]) >= p[u, k]
                        model += (BigM + p[i, k]) * (1 - Z[i][u][k]) + (s[u][k] - s[i][k]) >= p[i, k]

    # 3. Restrição do Makespan (Cmax)
    for i in N:
        model += lpSum(r[i, m_len - 1, k] * (s[i][k] + p[i, k]) for k in M) <= Cmax


    model.solve(PULP_CBC_CMD(msg=False))

    # Resultado
    #print(f"Status da Solução: {model.status} ({pulp.LpStatus[model.status]})")
    print(f"Makespan Mínimo (Cmax): {value(Cmax)}")


if __name__ == "__main__":
    n, m, p, r, bigM = read_instance('abz5.txt')

    print("Valor anterir:\n p:", p, "\n r:", r)

    manne_modeling(n, m, p, r, bigM)