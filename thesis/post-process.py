import sys

# remove fences
mdfile = sys.argv[1]

with open(mdfile, "r") as f:
    txt = f.readlines()

correct_refs = [l for l in txt if not l.startswith(":::")]

out = []
for l in  correct_refs:
    i = l.find("{#")
    if i == -1:
        out.append(l)
    else:
        out.append(l[:i-1]+"\n")

with open(mdfile, "w") as f:
    f.write("".join(out))
