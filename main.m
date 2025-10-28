% -------------------------------------------------------------------------
% Project: Wind Turbine Performance Modeling (ME 4053)
% File: main.m
% Authors: Aiden Wang, William Brandtjen, Christopher Idrovo Coronel
% Date: [Fill When Finished]
%
% Citation: 
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

% Remove 'CIRCLE'
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
% Given Parameter:
%   Wind Speed 9.9 [m/s]
%   Tip Speed Ratio 7.62 [-]

V_D2      = 9.9;          % [m/s]
lambda_D2 = 7.62;         % [-] TSR
beta_vec  = (-2:0.3:20)'; % [deg]   sweep range

% Convert lambda to rpm
rpm_D2 = lambda_D2 * V_D2 / (2*pi*cfg.R) * 60; % [rpm]

% Sweep β
CP = nan(size(beta_vec));   % temporary store as NAN
CT = CP;  P = CP;  Thrust = CP;  TSR = CP;
for k = 1:numel(beta_vec)
    o = bem_performance(cfg, blade, polars, V_D2, rpm_D2, beta_vec(k));
    CP(k)     = o.CP;
    CT(k)     = o.CT;
    P(k)      = o.P;
    Thrust(k) = o.Thrust;
    TSR(k)    = o.lambda;
end

% Locate optimum point
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
% Given Parameter 
%   Wind Speed = 7.5 [m/s]

V_D3 = 7.5;  % [m/s]
lambdaVec = (4:0.25:10)';      % TSR sweep
betaVec   = (-2:0.7:20)';        % deg sweep

CPmap   = nan(numel(betaVec), numel(lambdaVec));
CTmap   = CPmap;              
Pmap    = CPmap;              
RPMmap  = CPmap;                % rpm used for each lambda

for j = 1:numel(lambdaVec)
    rpm_j = lambdaVec(j) * V_D3 / (2*pi*cfg.R) * 60;  % TSR -> rpm
    RPMmap(:,j) = rpm_j;
    for i = 1:numel(betaVec)
        o = bem_performance(cfg, blade, polars, V_D3, rpm_j, betaVec(i));
        CPmap(i,j) = o.CP;
        CTmap(i,j) = o.CT;
        Pmap(i,j)  = o.P;
    end
end

% Locate global maximum CP and its (lambda*, beta*)
[CP_max, lin] = max(CPmap(:));
[iMax, jMax]  = ind2sub(size(CPmap), lin);
beta_star     = betaVec(iMax);
lambda_star   = lambdaVec(jMax);
rpm_star      = RPMmap(iMax, jMax);

% Save grids
T = array2table(CPmap,'VariableNames', compose('lambda_%.2f', lambdaVec), ...
'RowNames', compose('beta_%.2f', betaVec));
writetable(T, fullfile('outputs','D3_CP_map_lambda_beta.csv'), ...
    'WriteRowNames', true);

% C_P heatmap
fig = figure('Color','w','Position',[100 100 720 800]);
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

nexttile;
imagesc(lambdaVec, betaVec, CPmap); 
axis xy; 
colormap(parula);                        
cb = colorbar;
cb.Label.String = 'Power Coefficient, C_P [-]';
cb.Label.FontSize = 11;

hold on;
plot(lambda_star, beta_star, 'ko', 'MarkerSize',7, 'LineWidth',1.2);
text(lambda_star+0.1, beta_star, ...
    sprintf('\\lambda^*=%.2f,  \\beta^*=%.2f^\\circ,  C_P=%.3f', ...
            lambda_star, beta_star, CP_max), ...
    'FontSize',10,'FontAngle','italic','Color','b','VerticalAlignment','bottom');

xlabel('\lambda (Tip-Speed Ratio)','FontSize',11);
ylabel('\beta (Pitch Angle, deg)','FontSize',11);
title(sprintf('D3: Power Coefficient Map  C_P(\\lambda,\\beta) @ V=%.1f m/s', V_D3), ...
      'FontSize',12,'FontWeight','bold');
grid on; box on;

% Contour plot of C_P
nexttile;
[C,h] = contour(lambdaVec, betaVec, CPmap, 0:0.05:0.5, ...
                'LineWidth',1.0,'ShowText','on');
clabel(C,h,'FontSize',8,'Color',[0.1 0.1 0.1]);
colormap(parula);

hold on;
plot(lambda_star, beta_star, 'ko', 'MarkerFaceColor','k', 'MarkerSize',6);

xlabel('\lambda (Tip-Speed Ratio)','FontSize',11);
ylabel('\beta (Pitch Angle, deg)','FontSize',11);
title(sprintf('C_P Contours with Global Optimum  (\\lambda^*=%.2f, \\beta^*=%.2f^\\circ)', ...
               lambda_star, beta_star), ...
      'FontSize',12,'FontWeight','bold');

grid on; box on;

% Add legend 
legend({'Global Optimum Point'}, ...
       'Location','southwest','FontSize',9,'Box','off');
% For layout
set(findall(fig,'-property','FontName'),'FontName','Helvetica');
sgtitle('Phase 3 — Power Coefficient Map and Contours', ...
        'FontSize',13,'FontWeight','bold');
saveas(fig, fullfile('outputs','D3_CP_map_professional.png'));

% Console summary
fprintf('\n[D3] V=%.1f m/s: CP_max=%.3f at lambda*=%.2f, beta*=%.2f deg (rpm=%.2f)\n', ...
        V_D3, CP_max, lambda_star, beta_star, rpm_star);


%% Deliverable 4 
% Given Parameter 
%   Wind Speed = 18.8 [m/s]

V_D4 = 18.8;          % m/s
rpm_D4 = 15.5;        % rpm  (max rotational velocity)
Pcap  = cfg.P_rated;  % W    (2.5 MW)

% Power error function versus pitch angle (deg)
fP = @(beta) bem_performance(cfg, blade, polars, V_D4, rpm_D4, beta).P - Pcap;

% Solve for beta in a practical range
[beta_star, hit] = d4_find_beta(fP, 0, 30);
if ~hit
    [beta_star, hit] = d4_find_beta(fP, 30, 45);
    if ~hit
        warning('D4: No exact root found; using clipped beta = %.2f deg', beta_star);
    end
end

% Evaluate the final operating point
outD4 = bem_performance(cfg, blade, polars, V_D4, rpm_D4, beta_star);

% Console summary
fprintf('\n[D4] V=%.1f m/s, rpm=%.1f -> beta* = %.2f deg\n', V_D4, rpm_D4, beta_star);
fprintf('     P=%.3f MW (target 2.500), CP=%.3f, CT=%.3f, Thrust=%.0f kN, TSR=%.2f\n',...
        outD4.P/1e6, outD4.CP, outD4.CT, outD4.Thrust/1e3, outD4.lambda);

% Save CSV for the report
T_D4 = struct2table(struct( ...
    'V_mps', V_D4, 'rpm', rpm_D4, 'beta_star_deg', beta_star, ...
    'P_W', outD4.P, 'CP', outD4.CP, 'CT', outD4.CT, ...
    'Thrust_N', outD4.Thrust, 'lambda', outD4.lambda ));
writetable(T_D4, fullfile('outputs','D4_pitch_singlepoint.csv'));

% P vs beta showing the 2.5 MW crossing
beta_grid = (0:0.5:30)';
Pgrid = arrayfun(@(b) bem_performance(cfg, blade, polars, V_D4, rpm_D4, b).P, beta_grid);
fig = figure('Color','w');
plot(beta_grid, Pgrid/1e6, 'LineWidth',1.8); grid on; hold on;
yline(Pcap/1e6,'--','2.5 MW cap');
xline(beta_star,'k:','\beta^*');
xlabel('\beta (deg)'); ylabel('Power (MW)');
title(sprintf('D4: P(\\beta) @ V=%.1f m/s, rpm=%.1f', V_D4, rpm_D4));
saveas(fig, fullfile('outputs','D4_power_vs_beta_sanity.png'));



%% Deliverable 5
% Uses D4 thrust (tip load) + distributed tower drag q(z) via Cd(Re)

% Load D4 case (thrust and operating point)
if exist('outD4','var')
    Thrust_N = outD4.Thrust;
    V_case   = outD4.V;
    rpm_case = outD4.rpm;
    beta_case= outD4.beta_deg;
else
    Td4 = readtable(fullfile('outputs','D4_pitch_singlepoint.csv'));
    Thrust_N = Td4.Thrust_N(1);
    V_case   = Td4.V_mps(1);
    rpm_case = Td4.rpm(1);
    beta_case= Td4.beta_star_deg(1);
end

% Geometry & section properties
z   = tower.z_m(:);                   % [m]
OD  = tower.OD_m(:);                  % [m]
t   = tower.t_m(:);                   % [m]
ID  = max(OD - 2*t, 0);               % [m]
I   = (pi/64).*(OD.^4 - ID.^4);       % [m^4]
E   = cfg.steel_E;                    % [Pa]
Hhub= cfg.H_hub;                      % [m]

% Fine grid for integration
z_lin  = linspace(0, max(z), 600).';
OD_lin = interp1(z, OD, z_lin, 'linear', 'extrap');
ID_lin = max(OD_lin - 2*interp1(z, t, z_lin, 'linear', 'extrap'), 0);
I_lin  = (pi/64).*(OD_lin.^4 - ID_lin.^4);

% Distributed tower drag q(z) using cylinderCD(Re)
alpha   = 0.14;                 % typical onshore wind-shear exponent
Vref    = V_case;               
zref    = Hhub;
rho     = cfg.rho;
mu      = cfg.air_viscosity;

V_z = Vref * max(z_lin,1e-6).^alpha / (zref^alpha);      % local wind [m/s]
Re_z = rho .* V_z .* OD_lin ./ mu;                       % Reynolds number
Cd_z = arrayfun(@cylinderCD, Re_z);                      % drag coeff per height
q_z  = 0.5 .* rho .* (V_z.^2) .* Cd_z .* OD_lin;         % [N/m] distributed drag

% Bending moment diagram: tip load + distributed q(z)
Q1 = cumtrapz(z_lin, q_z);             
Q2 = cumtrapz(z_lin, Q1);              
M_distrib = (Q2(end) - Q2) - z_lin.*(Q1(end) - Q1);
M_tip = Thrust_N * max(Hhub - z_lin, 0);                 % tip load
M_lin = M_tip + M_distrib;                               % total moment [N·m]

% Deflection via Euler-Bernoulli (variable EI)
kappa = M_lin ./ (E .* I_lin);          
theta = cumtrapz(z_lin, kappa);         % slope
ydef  = cumtrapz(z_lin, theta);         % deflection [m]
delta_top = ydef(numel(ydef));          % top deflection [m]

% Base stress & safety factor
sigma_z = (M_lin .* (OD_lin/2)) ./ I_lin;    % stress distribution [Pa]
[maxStress, idxMax] = max(sigma_z);
M_base   = M_lin(1);
sigma_b  = sigma_z(1);
SF_yield = cfg.steel_sigma_y / maxStress;

% Print summary results
fprintf('\n================== DELIVERABLE 5 ==================\n');
fprintf('Operating case: V = %.2f m/s, rpm = %.2f, beta = %.2f°\n', V_case, rpm_case, beta_case);
fprintf('------------------------------------------------------------\n');
fprintf('Rotor thrust (tip load)          : %8.0f N (%.0f kN)\n', Thrust_N, Thrust_N/1e3);
fprintf('Integrated tower drag             : %8.0f N (%.0f kN)\n', trapz(z_lin,q_z), trapz(z_lin,q_z)/1e3);
fprintf('------------------------------------------------------------\n');
fprintf('Base bending moment               : %8.2f MN·m\n', M_base/1e6);
fprintf('Base bending stress               : %8.1f MPa\n', sigma_b/1e6);
fprintf('Maximum bending stress             : %8.1f MPa  @ z = %.2f m\n', maxStress/1e6, z_lin(idxMax));
fprintf('Yield strength (steel)             : %8.0f MPa\n', cfg.steel_sigma_y/1e6);
fprintf('Static safety factor (yield)       : %8.2f\n', SF_yield);
fprintf('------------------------------------------------------------\n');
fprintf('Top deflection (lateral)           : %8.3f m\n', delta_top);
fprintf('============================================================\n\n');

% Save results to CSV 
D5_summary = table( ...
    V_case, rpm_case, beta_case, Thrust_N, trapz(z_lin,q_z), ...
    M_base, sigma_b, maxStress, z_lin(idxMax), SF_yield, delta_top, ...
    'VariableNames', {'V_mps','rpm','beta_deg','Thrust_N','TowerDrag_N', ...
                      'BaseMoment_Nm','BaseStress_Pa','MaxStress_Pa','MaxStressLoc_m', ...
                      'SF_yield','TopDeflection_m'});
writetable(D5_summary, fullfile('outputs','D5_summary.csv'));

% Plots
fig = figure('Color','w');

subplot(3,1,1);
plot(z_lin, q_z, 'LineWidth',1.8); 
ylabel('q(z) [N/m]');
title('Distributed tower drag');

subplot(3,1,2);
plot(z_lin, M_lin/1e6, 'LineWidth',1.8);
ylabel('M(z) [MN·m]');
title('Bending moment (tip load + tower drag)');

subplot(3,1,3);
plot(z_lin, ydef, 'LineWidth',1.8); 
xlabel('Height z (m)');
ylabel('y(z) [m]');
title('Lateral deflection');

saveas(fig, fullfile('outputs','D5_tower_drag_moment_deflection.png'));










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

        % use momentum equations to estimate a new a
        a_new  = 1 / ( (4*F*sin(phi)^2) / (sigma*Cn) + 1 );
        ap_new = 1 / ( (4*F*sin(phi)*cos(phi)) / (sigma*Ct) - 1 );

        % Glauert correction for high induction a > 0.33
        a_new = glauert_correction(a_new, sigma, Cn, F, phi);

        % Help with convergence, increase stability
        relax = 0.25;
        if ~isfinite(a_new), a_new = a; end
        if ~isfinite(ap_new), ap_new = ap; end
        a  = (1-relax)*a  + relax*a_new;
        ap = (1-relax)*ap + relax*ap_new;

        % Convergence
        if max(abs([a-a_new, ap-ap_new])) < 1e-4, break; end
    end

    % Elemental forces (per blade) resolved to thrust/tangential
    qrel = 0.5 * rho * Vrel^2;      % dynamic pressure at this section
    dL   = qrel * ci * CL * dr(i);  % lift over annular strip (per blade)
    dD   = qrel * ci * CD * dr(i);  % drag over annular strip (per blade)
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

function [beta_star, hit] = d4_find_beta(fP, beta_lo, beta_hi)
% Find beta in [beta_lo, beta_hi] such that fP(beta)=0 (P≈Pcap).
P_lo = fP(beta_lo);  P_hi = fP(beta_hi);
hit = false;

if isfinite(P_lo) && isfinite(P_hi) && sign(P_lo)*sign(P_hi) <= 0
    try
        beta_star = fzero(fP, [beta_lo, beta_hi]);
        hit = true;  return;
    catch
        % fall through to clipping
    end
end

% No sign change → clip toward satisfying P ≤ cap (if possible),
% else pick the nearer boundary by absolute error.
if P_lo <= 0 && isfinite(P_lo)
    beta_star = beta_lo;
elseif P_hi <= 0 && isfinite(P_hi)
    beta_star = beta_hi;
else
    beta_star = (abs(P_lo) <= abs(P_hi)) * beta_lo + ...
                (abs(P_lo)  >  abs(P_hi)) * beta_hi;
end
end


function [C_D] = cylinderCD(Re)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION: cylinderCD(Re)
% PURPOSE : Computes drag coefficient of a cylinder in cross-flow.
%
% NOTE: This function was provided as part of [ME 4053].
%       Original code structure credited to course materials/instructor.
%       Reviewed and used by authors for project implementation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Re < 2*10^5
C_D = 11 * Re.^(-0.75) + 0.9 * (1.0 - exp(-1000./Re))...
+ 1.2 * (1.0 -exp(-(Re./4500).^0.7));
elseif Re <= 5*10^5
C_D = 10.^(0.32*tanh(44.4504 - 8 * log10 (Re))-0.238793158);
else
C_D = 0.1 * log10(Re) - 0.2533429;
end
end
