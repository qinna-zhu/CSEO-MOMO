%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% basic information about the problem
myFN = @forrester;  % this could be any user-defined function
designspace = [0;   % lower bound
               1];  % upper bound

ndv = length(designspace(1,:));

% create DOE
npoints = 5;
X = linspace(designspace(1), designspace(2), npoints)';
Y = feval(myFN, X);

% create test points
npointstest = 101;
Xtest = linspace(designspace(1), designspace(2), npointstest)';
Ytest = feval(myFN, Xtest);
%
%                SRGT: 'RBF'
%                   P: [5x1 double]
%                   T: [5x1 double]
%            RBF_type: 'MQ'
%               RBF_c: 2
%     RBF_usePolyPart: 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fit surrogate
polynomial = @(x) [1,x];

%     srgtOPT.P = P;
%         srgtOPT.T = T;   
%         srgtOPT.FIT_Fn     = FIT_Fn;
%         srgtOPT.RBF_type        = RBF_type;
%         srgtOPT.RBF_c           = RBF_c;
%         srgtOPT.RBF_usePolyPart = RBF_usePolyPart;
%srgtsRBFSetOptions(P, T, FIT_Fn, FIT_LossFn, RBF_type, RBF_c, RBF_usePolyPart)
 srgtOPT  = srgtsRBFSetOptions(X, Y, @rbf_build,'', 'MQ', 1 ,2)
 % srgtOPT  = srgtsRBFSetOptions(X, Y)
 %:creates a structure with each of the specified  fields.
%srgtOPT  = srgtsRBFSetOptions(X, Y, '@rbf_build','MQ',RBF_c=2,RBF_usePolyPart=(1,x));
srgtSRGT = srgtsRBFFit(srgtOPT);

Yhat    = srgtsRBFEvaluate(Xtest, srgtSRGT);

CRITERIA = srgtsErrorAnalysis(srgtOPT, srgtSRGT, Ytest, Yhat)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plots
figure(1); clf(1);
plot(X, Y, 'ok', ...
    Xtest, Ytest, '--k', ...
    Xtest, Yhat, '-b');
