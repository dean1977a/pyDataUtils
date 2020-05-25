#对应评级区间
def level(score):
    level = 0
    if score <= 600:
        level = "D"
    elif score <= 640 and score > 600 : 
        level = "C"
    elif score <= 680 and score > 640:
        level = "B"
    elif  score > 680 :
        level = "A"
    return level



#val['level'] = val.score.map(lambda x : level(x) )

#val.level.groupby(val.level).count()/len(val)