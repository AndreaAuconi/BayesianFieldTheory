ClearAll[t, tp, k, kPrime, G, H, I1InnerFunc, I1OuterIntegrand, I1Unsimplified, FUnsimplified, F, GrandIntegrandFunc, IntermediateResultFunc, FinalResult, Gk, Gkp];

allAssumptions = {Element[k, Reals], Element[kPrime, Reals]};

G[x_] := Sqrt[2 + x^2];
H[x_] := 1 / ((1 + x^2) * G[x]);

rulesIn = {G[k] -> Gk, G[kPrime] -> Gkp};
rulesOut = {Gk -> G[k], Gkp -> G[kPrime]};


I1InnerFunc[t_, k_, kPrime_] = Integrate[
    tp * Cos[kPrime * tp] * Exp[-tp * G[kPrime]],
    {tp, 0, t}, 
    Assumptions -> allAssumptions
];

I1OuterIntegrand[t_, k_, kPrime_] = I1InnerFunc[t, k, kPrime] * Cos[k * t] * Exp[-t * G[k]];

I1[k_, kPrime_] = Integrate[
    I1OuterIntegrand[t, k, kPrime],
    {t, 0, Infinity},
    Assumptions -> allAssumptions
];

F[k_, kPrime_] = FullSimplify[I1[k, kPrime] /. rulesIn] /. rulesOut;

Print["--- F[k_, kPrime_] after integration wrt dt, dt' ---"];
Print[F[k, kPrime]];

GrandIntegrandFunc[k_, kPrime_] = H[k] * H[kPrime] * F[k, kPrime];

IntermediateResultFunc[k_] = Integrate[
    GrandIntegrandFunc[k, kPrime], 
    {kPrime, -Infinity, Infinity},
    Assumptions -> allAssumptions
];

Print["--- Result after integrating w.r.t. k' ---"];
Print[IntermediateResultFunc[k]];

(* FIX: Use pattern matching to strip the ConditionalExpression. *)
IntermediateIntegrandFix = IntermediateResultFunc[k] //. {
    ConditionalExpression[expr_, cond_] :> expr
};

FinalResult = Integrate[
    IntermediateIntegrandFix, 
    {k, -Infinity, Infinity},
    Assumptions -> allAssumptions
];

Print["--- Final integration w.r.t. k ---"];
Print[FinalResult];

expressionString = ToString[FinalResult, InputForm];
Export["J_correction.txt", expressionString, "Text"];


