parameters = {t, u, k, kp};

integrandSpace = Cos[k*t] * Cos[kp*t] * Exp[t^2/(4*u)];

integralSpace = Integrate[integrandSpace, {t, -Infinity, Infinity},
    Assumptions -> {Element[u | k | kp, Reals], u < 0}];

Print["---integralSpace-----"]
Print[integralSpace]


integrandTime = integralSpace * Exp[(3 + k^2 + kp^2)*u] / Sqrt[-u];
integrandTime = FullSimplify[integrandTime]

integralTime = Integrate[integrandTime, {u, -Infinity, 0},
    Assumptions -> {Element[k | kp, Reals]}];
integralTime = FullSimplify[integralTime]

Print["---integralTime-----"]
Print[integralTime]


integrandSpectrumFirst = integralTime / (2 * Pi^(5/2) * (1 + k^2) * (1 + kp^2))
integrandSpectrumFirst = FullSimplify[integrandSpectrumFirst]

integralSpectrumFirst = Integrate[integrandSpectrumFirst, {kp, -Infinity, Infinity},
	Assumptions -> {Element[k, Reals]}];
integralSpectrumFirst = FullSimplify[integralSpectrumFirst]

Print["----integralSpectrumFirst----"]
Print[integralSpectrumFirst]

integralSpectrumSecond = Integrate[integralSpectrumFirst, {k, -Infinity, Infinity}];
integralSpectrumSecond = FullSimplify[integralSpectrumSecond]

Print["----integralSpectrumSecond----"]
Print[integralSpectrumSecond]

expressionString = ToString[integralSpectrumSecond, InputForm];
Export["GC2.txt", expressionString, "Text"];

