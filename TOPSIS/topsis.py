import pandas as pd
import numpy as np
import math as m 
import numbers as num

def norm(a):
    sqsum = 0
    for i in range(len(a)):
        sqsum += a.iloc[i]**2
    a = a/m.sqrt(sqsum)
    return a

def normall(data):
    check = data.isna()
    for i in range(len(check)):
        row = check.iloc[i]
        for j in range(len(row)):
            if row.iloc[j] or not isinstance(data.iloc[i,j],num.Number):
                raise ValueError(f"error at place ({i},{j}) in input data, value either not numeric or missing ")
    temp = data.T
    for i in range(len(temp)):
        temp.iloc[i,:] = norm(temp.iloc[i,:])
    
    data = temp.T
    return data

def topsis(d, w1, inc1):
    print("Normalization started")
    d = normall(d)
    print("Normalization complete")
    colnames = np.array(d.columns)
    w = w1.split(",")
    inc = inc1.split(",")
    wl = len(w)
    cal = d.T
    dl = len(cal)
    if wl != dl:
        raise IndexError(f"un-even matrices passed, columns of data: {dl} and number of weights passed: {wl}")
    if len(inc) != dl:
        raise IndexError(f"un-even matrices passed, columns of data: {dl} and number of inclinations passed: {wl}")
    
    best = []
    worst = []
    for i in range(dl):
        cal.iloc[i,:] = cal.iloc[i,:]*float(w[i])
        if inc[i] == "+":
            best.append(max(cal.iloc[i,:]))
            worst.append(min(cal.iloc[i,:]))
        elif inc[i] == "-":
            best.append(min(cal.iloc[i,:]))
            worst.append(max(cal.iloc[i,:]))
        else:
            raise ValueError(f"""leave no separation between input impacts, example if acceptable input: "+,-,+,-", example of wrong input: "+, - ,+ ,-" """)
    
    d = cal.T
    distp = []
    distn = []
    for i in range(len(d)):
        b = 0
        w0 = 0
        for j in range(dl):
            b += (d.iloc[i,j] - best[j])**2
            w0 += (d.iloc[i,j] - worst[j])**2
        distp.append(m.sqrt(b))
        distn.append(m.sqrt(w0))
    
    p = []
    for i in range(len(distn)):
        p.append(distn[i] / (distn[i] + distp[i]))
    
    ret = []
    i = 0
    for row in range(len(d)):
        ret.append([*np.array(d.iloc[row,:]), p[i]])
        i += 1
    
    colnames = np.append(colnames, "TOPSIS score")
    d = pd.DataFrame(ret, columns=colnames)
    return p

def run(inp_path, weights, impacts, outpath):
    try:
        data = pd.read_csv(inp_path)
        print(f"Read input file: {inp_path}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    d = data.drop(data.columns[0], axis=1)
    pscore = topsis(d, weights, impacts)
    
    data["Topsis Score"] = pscore
    data['Rank'] = data['Topsis Score'].rank(method='max', ascending=False).astype(int)
    
    try:
        data.to_csv(outpath, index=False)
        print(f"Saved results to: {outpath}")
    except Exception as e:
        print(f"Error saving result file: {e}")
        return


