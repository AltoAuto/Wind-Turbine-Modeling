% -------------------------------------------------------------------------
% Project: Wind Turbine Performance Modeling (ME 4053)
% File: main.m
% Authors: Aiden Wang, William Brandtjen, Christopher Idrovo Coronel
% Date: [Fill When Finished]
%
% Note on AI Assistance:
%   This code was fully authored by the authors.
%   ChatGPT was used as a brainstorming and code-review tool 
%   to help debugging, organize comments, improve readability, and check 
%   formatting. All modeling equations, logic, and implementation 
%   were selected and coded by the authors.
% -------------------------------------------------------------------------


clear; clc; close all;
if ~exist('outputs','dir')
    mkdir('outputs');
end

%% Define Parameter
cfg = struct();
cfg.rho      = 1.1;    % [kg/m^3] air density
cfg.air_viscosity   = 1.8e-5;     % [N·s/m^2] air viscocity 
cfg.R        = 48;       % [m] Blade/Rotor Radius
cfg.H_hub    = 80.4;     % [m] Hub Height
cfg.P_rated  = 2.5e6;    % [W] Rated Power
cfg.V_team   = 18.8;     % [m/s] Our Wind Speed
cfg.B        = 3;       % num blades
cfg.steel_density   = 7850;       % [kg/m^3] steel density 
cfg.steel_E         = 210e9;      % [Pa] Young's modulus, typical for steel
cfg.steel_sigma_u   = 450e6;      % [Pa] ultimate tensile
cfg.steel_sigma_y   = 345e6;      % [Pa] yield
cfg.steel_grade     = "ASTM A572 Grade 50";

% Blade geometry
T = readtable("data\blade geometry.csv");
T{:,1:3} = T{:,1:3} * 1e-3;   % first 3 columns in mm
T.Properties.VariableNames = {'r_m','dist_face','chord_m','twist_deg','airfoil_id'};
blade = T;

% Clean and normalize airfoil IDs
blade.airfoil_id = string(blade.airfoil_id);
blade.airfoil_id = upper(strtrim(blade.airfoil_id));

% Remove non-aerodynamic root sections labeled 'CIRCLE'
blade = blade(~strcmp(blade.airfoil_id, "CIRCLE"), :);

% Keep only the base airfoil name (remove everything after first hyphen)
for i = 1:height(blade)
    id = blade.airfoil_id(i);
    parts = regexp(id, 'DU\s*([0-9]+)', 'tokens', 'once');
    if ~isempty(parts)
        blade.airfoil_id(i) = "DU" + parts{1};
    end
end

% Load Airfoil for BEM
airfoil_names = {'DU91','DU93','DU96','DU97'};
polars = containers.Map;
for i = 1:numel(airfoil_names)
    name = airfoil_names{i};
    T = readtable(fullfile('data', ['airfoil_' name '.csv']));
    polars(name) = T;
end

% Load Tower Specs
tower = readtable('data/tower specs.csv');
tower.Properties.VariableNames = {'height_mm','OD_mm','t_mm'};
tower.z_m  = tower.height_mm*1e-3;
tower.OD_m = tower.OD_mm*1e-3;
tower.t_m  = tower.t_mm*1e-3;
tower.ID_m = tower.OD_m - 2*tower.t_m;

%% Deliverable 1
V_D1 = 10;       % [m/s] wind velocity
rpm_D1  = 14;   % [rpm] rotational velocity
beta_D1 = 0;    % [deg] pitch angle

% Call BEM kernel
out = bem_performance(cfg, blade, polars, V_D1, rpm_D1, beta_D1);

% Print result
fprintf('\n[D1] Single point @ V=%.1f m/s, rpm=%.1f, beta=%.1f°\n', V_D1, rpm_D1, beta_D1);
fprintf('CP=%.4f, CT=%.4f, P=%.1f kW, Thrust=%.0f kN, TSR=%.2f\n', out.CP, out.CT, out.P/1e3, out.Thrust/1e3, out.lambda);
writetable(struct2table(out), fullfile('outputs','D1_single_point.csv'));
bar_save_singlepoint(out, fullfile('outputs','D1_single_point_bar.png'));

%% Deliverable 2
% Given parameter:
%   Wind Speed 9.9 [m/s]
%   Tip Speed Ratio 7.62 [-]

V_D2      = 9.9;          % m/s
lambda_D2 = 7.62;         % [-]
beta_vec  = (-2:0.5:20)'; % deg   sweep range

% Convert lambda to rpm
rpm_from_lambda = lambda_D2 * V_D2 / (2*pi*cfg.R) * 60;
rpm_D2 = rpm_from_lambda;

% Sweep β
CP = nan(size(beta_vec));
CT = CP;  P = CP;  Thrust = CP;  TSR = CP;
for k = 1:numel(beta_vec)
    o = bem_performance(cfg, blade, polars, V_D2, rpm_D2, beta_vec(k));
    CP(k)     = o.CP;
    CT(k)     = o.CT;
    P(k)      = o.P;
    Thrust(k) = o.Thrust;
    TSR(k)    = o.lambda;
end

% Locate optimum
[CP_max, idxMax] = max(CP);
beta_star = beta_vec(idxMax);

% Save table
T_D2 = table(beta_vec, CP, CT, P, Thrust, TSR, ...
    'VariableNames', {'beta_deg','CP','CT','P_W','Thrust_N','lambda'});
writetable(T_D2, fullfile('outputs','D2_CP_vs_pitch_fixedTSR.csv'));

% Plot CP vs beta with marker at optimum
fig = figure('Color','w');
plot(beta_vec, CP, 'LineWidth',1.8); grid on; hold on;
plot(beta_star, CP_max, 'o', 'MarkerSize',7, 'MarkerFaceColor',[.1 .1 .1]);
xlabel('\beta (deg)'); ylabel('C_P');
title(sprintf('D2: C_P vs Pitch @ V=%.1f m/s, \\lambda=%.2f (rpm=%.2f)', V_D2, lambda_D2, rpm_D2));
text(beta_star, CP_max, sprintf('  \\beta^* = %.2f^\\circ, C_P=%.3f', beta_star, CP_max), ...
    'VerticalAlignment','bottom');
saveas(fig, fullfile('outputs','D2_CP_vs_pitch_fixedTSR.png'));

% Console summary
fprintf('\n[D2] V=%.1f m/s, lambda=%.2f (rpm=%.2f): beta* = %.2f deg, CP_max = %.3f\n', ...
        V_D2, lambda_D2, rpm_D2, beta_star, CP_max);

%% Deliverable 3


%% LOCAL FUNCTION %% LOCAL FUNCTION %% LOCAL FUNCTION %% LOCAL FUNCTION %%

function out = bem_performance(cfg, blade, polars, V, rpm, beta_deg)
% BEM_PERFORMANCE  Evaluate wind turbine performance at a single operating point 
%                  using Blade Element Momentum (BEM) theory.
%
% DESCRIPTION:
%   Implements a BEM solver for one steady operating condition, defined by
%   wind speed, rotor speed, and collective blade pitch. The model combines
%   blade geometry with airfoil polars to compute spanwise forces, which are
%   then integrated to predict overall rotor performance metrics.
%
% INPUTS:
%   cfg      - Configuration structure containing rotor and air properties
%              (fields: rho [kg/m^3], R [m], B [# of blades], etc.)
%   blade    - Table/struct of blade geometry with fields:
%              r_m [m], chord_m [m], twist_deg [deg], airfoil_id [string]
%   polars   - Airfoil aerodynamic data (CL, CD vs. angle of attack, Re, etc.)
%   V        - Free-stream wind speed [m/s]
%   rpm      - Rotor rotational speed [rev/min]
%   beta_deg - Collective blade pitch angle [deg]
%
% OUTPUTS (struct 'out'):
%   CP       - Power coefficient, C_P [-]
%   CT       - Thrust coefficient, C_T [-]
%   P        - Shaft power [W]
%   Thrust   - Rotor thrust force [N]
%   lambda   - Tip speed ratio, TSR [-]
%   rpm      - Input rotor speed [rev/min]
%   V        - Input wind speed [m/s]
%   beta_deg - Input blade pitch [deg]
%
% NOTES:
%   - Assumes steady, axisymmetric inflow with BEM corrections (tip-loss,
%     high-induction, etc. if implemented).
%   - Results valid for the specified operating point only; use in a loop
%     for full performance maps.

rho   = cfg.rho;      % Air density [kg/m^3], taken from configuration (affects aerodynamic loads and power extraction)
R     = cfg.R;        % Rotor radius [m], defines the swept area and radial span of analysis
B     = cfg.B;        % Number of blades [-], influences solidity, aerodynamic loading, and tip-loss correction
omega = rpm * 2*pi/60; % Rotor angular velocity [rad/s], converted from rotational speed in RPM to SI units
A     = pi * R^2;     % Rotor swept area [m^2], used in performance coefficients (CP, CT) and wind power calculations

% create spanwise vectors and ensure they're sorted by radius
r = blade.r_m(:);
chord = blade.chord_m(:);
twist_deg = blade.twist_deg(:);
airfoil_id = string(blade.airfoil_id(:));
[~,ix] = sort(r); r=r(ix); chord=chord(ix); twist_deg=twist_deg(ix); airfoil_id=airfoil_id(ix);

% Discretization spacing, avoid singularities
dr = [diff(r); max(1e-3, r(end)-r(end-1))];   % last segment length approx
r_hub = max(0.05*cfg.R, r(1));                % avoid singular near hub

% Results accumulators
dQ = zeros(size(r));      % elemental torque
dT = zeros(size(r));      % elemental axial force

for i = 1:numel(r)
    ri = r(i);
    if ri < r_hub || ri > R, continue; end

    ci    = chord(i);
    thetai= twist_deg(i);
    af    = airfoil_id(i);

    % Init induction factors, reasonable for loaded rotor
    a  = 0.30;
    ap = 0.0;

    % Iteration settings
    for it = 1:100
        % Kinematics
        Vax = V*(1 - a);
        Vtan = omega*ri*(1 + ap);
        Vrel = hypot(Vax, Vtan);
        phi  = atan2(Vax, Vtan);                 % rad (inflow angle)
        alpha = rad2deg(phi) - (thetai + beta_deg);  % deg (AoA)

        % Aerodynamics from polars (robust header handling)
        [CL, CD] = interp_polar(polars, af, alpha);

        % Normal/tangential coefficients wrt rotor plane
        Cn =  CL*cos(phi) + CD*sin(phi);
        Ct =  CL*sin(phi) - CD*cos(phi);

        % Solidity at this station
        sigma = B*ci / (2*pi*ri);

        % Prandtl tip loss
        F = prandtl_tiploss(B, R, ri, phi);

        % Momentum equations
        a_new  = 1 / ( (4*F*sin(phi)^2) / (sigma*Cn) + 1 );
        ap_new = 1 / ( (4*F*sin(phi)*cos(phi)) / (sigma*Ct) - 1 );

        % Glauert correction for high induction a > acrit
        a_new = glauert_correction(a_new, sigma, Cn, F, phi);

        % Relaxation to stabilize
        relax = 0.25;
        if ~isfinite(a_new), a_new = a; end
        if ~isfinite(ap_new), ap_new = ap; end
        a  = (1-relax)*a  + relax*a_new;
        ap = (1-relax)*ap + relax*ap_new;

        % Convergence
        if max(abs([a-a_new, ap-ap_new])) < 1e-4, break; end
    end

    % Elemental forces (per blade) resolved to thrust/tangential
    qrel = 0.5 * rho * Vrel^2;
    dL = qrel * ci * CL * dr(i);
    dD = qrel * ci * CD * dr(i);
    Fa = dL*cos(phi) + dD*sin(phi);     % axial per blade
    Ft = dL*sin(phi) - dD*cos(phi);     % tangential per blade

    dT(i) = B * Fa;                     % thrust (all blades)
    dQ(i) = B * Ft * ri;                % torque (all blades)
end

T_ax = sum(dT);
Q    = sum(dQ);
P    = Q * omega;

%Compute CP and CT
CP   = P / (0.5*rho*A*V^3);
CT   = T_ax / (0.5*rho*A*V^2);
lambda = omega*R / V;

out = struct('V',V,'rpm',rpm,'beta_deg',beta_deg, ...
    'CP',CP,'CT',CT,'P',P,'Thrust',T_ax,'lambda',lambda);
end

function [CL, CD] = interp_polar(polars, airfoil_id, alpha_deg)
% Interpolate CL/CD at given alpha for a specific airfoil (robust to headers)
T = polars(char(airfoil_id));
vn = lower(strrep(T.Properties.VariableNames,'_',''));
% try to find columns
iAoA = find(ismember(vn, {'aoa','alpha','angleofattack'}), 1);
iCL  = find(ismember(vn, {'cl'}), 1);
iCD  = find(ismember(vn, {'cd'}), 1);
AoA  = T{:,iAoA}; CLtab = T{:,iCL}; CDtab = T{:,iCD};

% clamp outside range
alpha_deg = max(min(alpha_deg, max(AoA)), min(AoA));
CL = interp1(AoA, CLtab, alpha_deg, 'linear','extrap');
CD = interp1(AoA, CDtab, alpha_deg, 'linear','extrap');
end

function F = prandtl_tiploss(B, R, r, phi)
% Basic tip loss (no hub loss)
f = (B/2) * (R - r) ./ (r .* abs(sin(phi) + 1e-12));
F = (2/pi) * acos( exp(-f) );
F = max(min(F,1), 1e-3);
end

function a_corr = glauert_correction(a_raw, sigma, Cn, F, phi)
% Simple Glauert blend for a > ~0.33
ac = 0.33;
if a_raw <= ac
    a_corr = a_raw;
else
    % Using a classic empirical form
    CTloc = (4*F*(sin(phi))^2) * (a_raw./(1-a_raw)) * (sigma*Cn)/(4*F*(sin(phi))^2);
    % fall back if CTloc not well-defined
    if ~isfinite(CTloc) || CTloc<=0, CTloc = 4*F*ac*(1-ac); end
    a_corr = 0.5*(2 + CTloc) - sqrt( (0.5*(2 + CTloc))^2 - 1 );
    a_corr = min(max(a_corr, 0), 0.95);
end
end

function bar_save_singlepoint(out, filepath)
    fig = figure('Color','w'); 
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    nexttile;
    bar(out.CP); title('C_P'); ylim([0 max(0.7,out.CP*1.2)]); grid on;
    nexttile;
    bar(out.CT); title('C_T'); ylim([0 max(2.5,out.CT*1.2)]); grid on;
    nexttile;
    bar([out.P/1e6, out.Thrust/1e3]);
    set(gca,'XTickLabel',{'P [MW]','T [kN]'}); grid on;
    sgtitle(sprintf('D1: V=%.1f m/s, rpm=%.1f, \\beta=%.1f° , TSR=%.2f', out.V, out.rpm, out.beta_deg, out.lambda));
    saveas(fig, filepath);
end
