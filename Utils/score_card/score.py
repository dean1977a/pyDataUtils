def score(person_info,finance_info,credit_info,act_info):
    xbeta = person_info * ( 3.49460978) + finance_info * ( 11.40051582 ) + credit_info * (2.45541981) + act_info * ( -1.68676079) --0.34484897 
    score = 650-34* (xbeta)/math.log(2)
    return score
val['score'] = val.apply(lambda x : score(x.person_info,x.finance_info,x.credit_info,x.act_info) ,axis=1)

fpr_lr,tpr_lr,_ = roc_curve(val_y,val['score'])
val_ks = abs(fpr_lr - tpr_lr).max()
print('val_ks : ',val_ks)

