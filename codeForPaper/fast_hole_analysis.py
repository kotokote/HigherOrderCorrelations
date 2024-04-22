from numba import njit, prange

@njit
def rec(i, stack, stack_len, birth, death, out, max_length, G):
    if birth <= death:
        return
    if stack_len > 2 and G[i, stack[0]] > death:
        out.append((stack_len, min(birth, G[i, stack[0]]), death))
    if stack_len >= max_length:
        return
    if stack_len > 2:
        death = max(death, G[i, stack[0]])
    for j in range(stack[0] + 1, G.shape[0]):
        ok = True
        for k in range(stack_len):
            ok &= stack[k] != j
        if not ok:
            continue
        death1 = death
        for k in range(1, stack_len - 1):
            death1 = max(death1, G[j, stack[k]])
        stack[stack_len] = j
        rec(j, stack, stack_len + 1, min(birth, G[i, j]), death1, out, max_length, G)

@njit(parallel=True)
def fast_hole_analysis(G, max_length):
    ans = [(-1, 0.0, 0.0)]
    for i in range(G.shape[0]):
        if G.shape[0] > 400 and i % 10 == 0:
            print(i, len(ans))
        stack = [0] * max_length
        stack[0] = i
        rec(i, stack, 1, 1.0, 0.0, ans, max_length, G)
    return ans[1:]

def connected_components_analysis_range(G, r_min, r_max):
    edges = []
    for i in range(G.shape[0]):
        for j in range(i + 1, G.shape[0]):
            if G[i, j] >= r_min and G[i, j] < r_max:
                edges.append((i, j))
    p = list(range(G.shape[0]))
    r = [0] * G.shape[0]
    size = [1] * G.shape[0]
    def get(i):
        if p[i] == i:
            return i
        p[i] = get(p[i])
        return p[i]
    def unite(i, j):
        i = get(i)
        j = get(j)
        if i == j:
            return
        if r[i] < r[j]:
            size[j] += size[i]
            p[i] = j
        else:
            size[i] += size[j]
            p[j] = i
            if r[i] == r[j]:
                r[i] += 1
    for i, j in edges:
        unite(i, j)
    connected_components_counts = [size[i] for i in range(G.shape[0]) if p[i] == i]
    return connected_components_counts

def connected_components_analysis(G):
    edges = []
    for i in range(G.shape[0]):
        for j in range(i + 1, G.shape[0]):
            edges.append((G[i, j], i, j))
    edges.sort()
    edges = edges[::-1]
    connected_components_counts = [0] * (G.shape[0] + 1)
    connected_components_counts[1] = G.shape[0]
    p = list(range(G.shape[0]))
    r = [0] * G.shape[0]
    size = [1] * G.shape[0]
    has_cycle = [False] * G.shape[0]
    def get(i):
        if p[i] == i:
            return i
        p[i] = get(p[i])
        return p[i]
    def unite(i, j):
        i = get(i)
        j = get(j)
        if i == j:
            has_cycle[i] = True
            return
        if r[i] < r[j]:
            size[j] += size[i]
            p[i] = j
        else:
            size[i] += size[j]
            p[j] = i
            if r[i] == r[j]:
                r[i] += 1
    for _, i, j in edges:
        unite(i, j)
        root = get(i)
        if not has_cycle[root]:
            connected_components_counts[size[root]] += 1
    while connected_components_counts[-1] == 0:
        connected_components_counts.pop()
    return connected_components_counts