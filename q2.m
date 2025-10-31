ClearAll[t, s, sigma, alpha];

assumptions = {sigma > 0, alpha > 0, Sqrt[alpha] > (3/2) * sigma};
$Assumptions = assumptions;

IntegrandQ1[t_, s_] := Exp[-sigma * Sqrt[alpha] * (t + s)] * Exp[sigma^2 * Min[t, s]];
I1 = Integrate[
    IntegrandQ1[t, s], 
    {t, 0, Infinity}, 
    {s, 0, Infinity},
    Assumptions -> assumptions
];
Print["Integral over Q1 (I1): ", FullSimplify[I1]];

IntegrandQ2[t_, s_] := Exp[-sigma * Sqrt[alpha] * (-t + s)] * Exp[-sigma^2 * t];
I2 = Integrate[
    IntegrandQ2[t, s], 
    {t, -Infinity, 0}, 
    {s, 0, Infinity},
    Assumptions -> assumptions
];
Print["Integral over Q2 (I2): ", FullSimplify[I2]];

IntegrandQ3[t_, s_] := Exp[-sigma * Sqrt[alpha] * (-t - s)] * Exp[sigma^2 * (-t - s + Min[-t, -s])];
I3 = Integrate[
    IntegrandQ3[t, s], 
    {t, -Infinity, 0}, 
    {s, -Infinity, 0},
    Assumptions -> assumptions
];
Print["Integral over Q3 (I3): ", FullSimplify[I3]];

IntegrandQ4[t_, s_] := Exp[-sigma * Sqrt[alpha] * (t - s)] * Exp[-sigma^2 * s];
I4 = Integrate[
    IntegrandQ4[t, s], 
    {t, 0, Infinity}, 
    {s, -Infinity, 0},
    Assumptions -> assumptions
];
Print["Integral over Q4 (I4): ", FullSimplify[I4]];

TotalIntegral = I1 + I2 + I3 + I4;
TotalIntegralSimplified = FullSimplify[TotalIntegral, Assumptions -> assumptions];

Print["Total Integral I = I1 + I2 + I3 + I4:"];
Print[TotalIntegralSimplified];

FinalResult = (sigma^2 * alpha / 4) * TotalIntegralSimplified;

Print["Result:"];
FinalResultSimplified = FullSimplify[FinalResult, Assumptions -> assumptions];
Print[FinalResultSimplified]

expressionString = ToString[FinalResultSimplified, InputForm];
Export["q2.txt", expressionString, "Text"];

