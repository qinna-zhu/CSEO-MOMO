function [yhat] = my_srgtsGPPredictor(x, srgtSRGT)
%Function srgtsGPPredictor returns the predicted response and the
%estimated prediction variance of a Gaussian process model. Thus, for
%example:
%
%     [YHAT PREDVAR] = srgtsGPPredictor(X, SURROGATE): returns both the
%     predicted response YHAT and the prediction variance PREDVAR of the
%     Gaussian process model SURROGATE at all X sites. X can be either a single row
%     vector (single point, with each column representing a variable) or a
%     matrix (each row represents a point). YHAT and PREDVAR are
%     NPOINTS-by-1 vectors.
%
%Example:
%     % basic information about the problem
%     myFN = @cos;  % this could be any user-defined function
%     designspace = [0;     % lower bound
%                    2*pi]; % upper bound
%
%     % create DOE
%     npoints = 5;
%     X = linspace(designspace(1), designspace(2), npoints)';
%
%     % evaluate analysis function at X points
%     Y = feval(myFN, X);
%
%     % fit surrogate models
%     options = srgtsGPSetOptions(X, Y);
% 
%     [surrogate state] = srgtsGPFit(options);
%
%     % create test points
%     Xtest = linspace(designspace(1), designspace(2), 100)';
%
%     % evaluate surrogate at Xtest
%     [Yhat PredVar] = srgtsGPPredictor(Xtest, surrogate);
%
%     plot(X, Y, 'o', ...
%          Xtest, Yhat, ...
%          Xtest, Yhat + sqrt(PredVar), 'r', ...
%          Xtest, Yhat - sqrt(PredVar), 'r')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Felipe A. C. Viana
% felipeacviana@gmail.com
% http://sites.google.com/site/felipeacviana
%
% This program is free software; you can redistribute it and/or
% modify it. This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% limitations on predictor function
[yhat predvar] = gpml_gpr_predictor(x, srgtSRGT.GP_CovarianceFunction, srgtSRGT.GP_LogTheta, srgtSRGT.P, srgtSRGT.T);
yhat =yhat';
return
