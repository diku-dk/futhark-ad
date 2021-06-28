
let dotproduct [n] (a: [n]f64) (b: [n]f64) : f64 =
    f64.sum <| map2 (*) a b

let sigmoid (x: f64) : f64 =
    1.0f64 / (1.0f64 + f64.exp(-x))

let lstmModel [d] (weight: [4][d]f64) (bias: [4][d]f64)
                  (hidden: [d]f64) (cell: [d]f64) (input: [d]f64)
                : ([d]f64, [d]f64) =
    let forget = map sigmoid <| map2 (+) bias[0] <| map2 (*) input  weight[0]
    let ingate = map sigmoid <| map2 (+) bias[1] <| map2 (*) hidden weight[1]
    let outgate= map sigmoid <| map2 (+) bias[2] <| map2 (*) input  weight[2]
    let change = map f64.tanh<| map2 (+) bias[3] <| map2 (*) hidden weight[3]

    let t1s = map2 (*) cell forget
    let t2s = map2 (*) ingate change
    let cell2 = map2 (+) t1s t2s

    let hidden2 = map2 (*) outgate <| map f64.tanh cell2
    in (hidden2, cell2)

let lstmPredict [slen] [d]
                (mainParams: [slen][2][4][d]f64)
                (extraParams: [3][d]f64)
                (state: [slen][2][d]f64)
                (input: [d]f64) : ([d]f64, [slen][2][d]f64) =
    let x0 = map2 (*) input extraParams[0]
    let state_ini = replicate slen <| replicate 2 <| replicate d 0f64 -- : [slen][2][d]f64
    let (state', x') =
        loop (s, x) = (state_ini, x0)
        for i < slen do
            let (h, c) = lstmModel mainParams[i,0] mainParams[i,1] state[i,0] state[i,1] x
            let i_rev = slen - i - 1
            let s[i_rev, 0] = c
            let s[i_rev, 1] = h
            in  (s, h)
    let v' = map2 (*) x' extraParams[1] |>
             map2 (+) extraParams[2]
    in  (v', state')

let lstmObjective   [stlenx2] [dx4] [lenSeq] [d]
                    (mainParams0: [stlenx2][dx4]f64)
                    (extraParams: [3][d]f64)
                    (state0: [stlenx2][d]f64)
                    (sequence: [lenSeq][d]f64) : f64 =
    let stlen = assert (0 == stlenx2 % 2 && dx4 == 4*d) (stlenx2 / 2)
    -- mainParams : [stlen][2][4][d]f64
    let mainParams = unflatten stlen 2 <| map (unflatten 4 d) mainParams0
    -- state : [stlen][2]f64
    let state = unflatten stlen 2 state0
    let (_, total) =
        loop (oldState, oldTotal) = (state, 0f64) for i < lenSeq - 1 do
            let (y_pred, newState) = lstmPredict mainParams extraParams oldState sequence[i] -- y_pred: DV [d]f64, newState: DM
            let tmp_sum = f64.sum <| map f64.exp y_pred
            let tmp_log = - f64.log (tmp_sum + 2.0f64)
            let ynorm = map (+tmp_log) y_pred
            let newTotal = oldTotal + (dotproduct sequence[i+1] ynorm)
            in  (newState, newTotal)
    let count = d * (lenSeq - 1)
    in - (total / f64.i64(count))

let main   [stlenx2] [dx4] [lenSeq] [d]
                    (mainParams0: [stlenx2][dx4]f64)
                    (extraParams: [3][d]f64)
                    (state0: [stlenx2][d]f64)
                    (sequence: [lenSeq][d]f64) : f64 =
    lstmObjective mainParams0 extraParams state0 sequence