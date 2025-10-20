parameters = {t, tp, u, up, k, kp};

integrandSpace = Cos[k*tp] * Cos[kp*(t - tp)] * Exp[(t^2/u + (t-tp)^2/(up-u))/4];

integralSpaceFirst = Integrate[integrandSpace, {tp, -Infinity, Infinity},
    Assumptions -> {Element[t | u | up | k | kp, Reals], u < 0, up < 0, up < u}];

Print["---integralSpaceFirst-----"]
Print[integralSpaceFirst]

integralSpaceSecond = Integrate[integralSpaceFirst, {t, -Infinity, Infinity},
    Assumptions -> {Element[u | up | k | kp, Reals], u < 0, up < 0, up < u}];

Print["---integralSpaceSecond-----"]
Print[integralSpaceSecond]

integrandTime = integralSpaceSecond * Exp[-u + (3 + k^2)*up -kp^2*(u-up)] / Sqrt[u*(up-u)];
integrandTime = FullSimplify[integrandTime]

integralTime = Integrate[integrandTime, {u, -Infinity, 0}, {up, -Infinity, u},
    Assumptions -> {Element[k | kp, Reals]}];
integralTime = FullSimplify[integralTime]

Print["---integralTime-----"]
Print[integralTime]

integrandSpectrumFirst = integralTime  / (2 * Pi^3 * (1 + k^2) * (1 + kp^2))
integrandSpectrumFirst = FullSimplify[integrandSpectrumFirst]

integralSpectrumFirst = Integrate[integrandSpectrumFirst, {kp, -Infinity, Infinity},
	Assumptions -> {Element[k, Reals]}];
integralSpectrumFirst = FullSimplify[integralSpectrumFirst]

Print["----integralSpectrumFirst----"]
Print[integralSpectrumFirst]

expressionString = ToString[integralSpectrumFirst, InputForm];
Export["GGCC.txt", expressionString, "Text"];

