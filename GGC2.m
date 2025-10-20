parameters = {t, tp, u, up, k, kp};

integrandSpace = Cos[k*(t - tp)] * Cos[kp*(t - tp)] * Exp[(t^2/u + tp^2/up)/4];

integralSpace = Integrate[integrandSpace, {t, -Infinity, Infinity}, {tp, -Infinity, Infinity},
    Assumptions -> {Element[u | up | k | kp, Reals], u < 0, up < 0}];

Print["---integralSpace-----"]
Print[integralSpace]


integrandTime = integralSpace * Exp[u + up -(2 + k^2 + kp^2)*Abs[u - up]] / Sqrt[u*up];
integrandTime = FullSimplify[integrandTime]

integralTime = Integrate[integrandTime, {u, -Infinity, 0}, {up, -Infinity, 0},
    Assumptions -> {Element[k | kp, Reals]}];
integralTime = FullSimplify[integralTime]

Print["---integralTime-----"]
Print[integralTime]


integrandSpectrumFirst = integralTime  / ((2*Pi)^3 * (1 + k^2) * (1 + kp^2))
integrandSpectrumFirst = FullSimplify[integrandSpectrumFirst]

integralSpectrumFirst = Integrate[integrandSpectrumFirst, {kp, -Infinity, Infinity},
	Assumptions -> {Element[k, Reals]}];
integralSpectrumFirst = FullSimplify[integralSpectrumFirst]

Print["----integralSpectrumFirst----"]
Print[integralSpectrumFirst]

expressionString = ToString[integralSpectrumFirst, InputForm];
Export["GGC2.txt", expressionString, "Text"];

