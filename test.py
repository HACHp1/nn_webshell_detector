# 五年所得

# prof_rate=0.058

prof_rate=0.0363

def five_year(x):
    return x*(1+prof_rate*5)


print(
    five_year(
        five_year(
            five_year(
                five_year(45))
        )
    )
)
