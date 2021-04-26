-- Expands a BA data file (e.g. ba1_n49_m7776_p31843.txt).  By
-- fortuitous circumstance, these files are actually valid Futhark
-- data files (in text format)!
--
-- Running:
--
-- $ ./expand -b < ba1_n49_m7776_p31843.txt > ba1_n49_m7776_p31843.in

let main n m p
         cams0 cams1 cams2 cams3 cams4 cams5 cams6 cams7 cams8 cams9 cams10
         X0 X1 X2
         w
         feats0 feats1 :
  ([n][11]f64, [m][3]f64, [p]f64, [p][2]i32, [p][2]f64) =
  (replicate n [cams0, cams1, cams2, cams3, cams4, cams5, cams6, cams7, cams8, cams9, cams10],
   replicate m [X0,X1,X2],
   replicate p w,
   tabulate p (\i -> [i32.i64 (i%n), i32.i64 (i%m)]),
   replicate p [feats0,feats1])
