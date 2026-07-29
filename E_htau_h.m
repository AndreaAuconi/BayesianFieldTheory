g[t_] := Exp[-Abs[tau - t]/tau] - Exp[-Abs[t]/tau];

Term1 = (r0 / (4 * tau^2)) * Integrate[
   g[t]^2, 
   {t, -Infinity, Infinity}, 
   Assumptions -> tau > 0
];

Term2 = (r0^2 * sigma^2 / (4 * tau^2)) * Integrate[
   g[t] * g[tp] * Boole[t * tp > 0] * Min[Abs[t], Abs[tp]],
   {t, -Infinity, Infinity},
   {tp, -Infinity, Infinity},
   Assumptions -> tau > 0
];

TheSum = Term1 + Term2;

FinalResult = FullSimplify[
  TheSum /. sigma -> 1 / (tau * Sqrt[r0]),
  Assumptions -> {tau > 0, r0 > 0}
];

Print["E[(h_tau - h_0)^2 | r_0] =   ", FinalResult];

Export["E_htau_h.txt", "E[(h_tau - h_0)^2 | r_0] = " <> ToString[TextForm[FinalResult]], "Text"];
