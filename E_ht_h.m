g[u_, t_] := Exp[-Abs[t - u]/tau] - Exp[-Abs[u]/tau];

Term1 = (r0 / (4 * tau^2)) * Integrate[
   g[u, t]^2, 
   {u, -Infinity, Infinity}, 
   Assumptions -> {tau > 0, t > 0}
];

Term2 = (r0^2 * sigma^2 / (4 * tau^2)) * Integrate[
   g[u, t] * g[v, t] * Boole[u * v > 0] * Min[Abs[u], Abs[v]],
   {u, -Infinity, Infinity},
   {v, -Infinity, Infinity},
   Assumptions -> {tau > 0, t > 0}
];

TheSum = Term1 + Term2;

FinalResult = FullSimplify[
  TheSum /. sigma -> 1 / (tau * Sqrt[r0]),
  Assumptions -> {tau > 0, t > 0, r0 > 0}
];

Print["E[(h_t - h_0)^2 | r_0] =   ", FinalResult];

Export["E_ht_h.txt", "E[(h_t - h_0)^2 | r_0] = " <> ToString[TextForm[FinalResult]], "Text"];

