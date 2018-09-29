# Convert Expression
def convertion_expression(value):
    expression={'AF':'Afraid','AN':'Angry','DI':'Disgusted',
                'HA':'Happy','NE':'Neutral','SA':'Sad','SU':'Surprised'}
    return expression[value]

#Convert Angle
def convertion_angle(value):
    angle={'S':'Straight','HR':'Half Right','HL':'Half Left','FR':'Right','FL':'Left'}
    return angle[value]

# save complete csv result
def csv_result(real_class,prediction,csv_,path):
    s=dict(zip(real_class["Straight"].index,prediction["Straight"]))
    hl=dict(zip(real_class["Half Left"].index,prediction["Half Left"]))
    hr=dict(zip(real_class["Half Right"].index,prediction["Half Right"]))
    l=dict(zip(real_class["Left"].index,prediction["Left"]))
    r=dict(zip(real_class["Right"].index,prediction["Right"]))
    final_pred={**s,**hl,**hr,**l,**r}
    final_pred_=list(final_pred.get(i) for i in sorted(final_pred.keys()))
    csv_['Encoding_expression'] = final_pred_
    csv_['Encoding_expression'] = csv_['Encoding_expression'].map({0:'Afraid', 1:'Angry',2: 'Disgusted', 3:'Happy', 4:'Neutral',5:'Sad',6:'Surprised'})
    csv_.to_csv(path, index=False)    

