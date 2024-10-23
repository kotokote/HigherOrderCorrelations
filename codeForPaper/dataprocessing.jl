using Ripserer

function run_ripserer(data; maxdim=3)
    r = ripserer(data; dim_max=maxdim, alg=:involuted)
    result = [[(interval.birth, interval.death, [[v - 1 for v in vertices(s[1])] for s in interval.representative]) for interval in r[i]] for i in 2:maxdim]
    return result
end