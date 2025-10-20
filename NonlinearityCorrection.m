text1 = Import["GGC2.txt", "Text"];
text2 = Import["GGCC.txt", "Text"];

expr1 = ToExpression[text1];
expr2 = ToExpression[text2];

summedExpression = expr1 + expr2;
summedExpression = FullSimplify[summedExpression]

Print["--- Summed Expression (GGC2 term + GGCC term) ---"];
Print[summedExpression];

integratedResult = Integrate[summedExpression, {k, -Infinity, Infinity}];

Print["--- Result of Integration w.r.t. k ---"];
Print[integratedResult];

expressionString = ToString[integratedResult, InputForm];
Export["NonlinearityCorrection.txt", expressionString, "Text"];
