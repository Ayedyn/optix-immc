%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  two-pass excitation/emission ICG cuvette fluorescence demo
%
%  Reproduces the ICG cuvette case of the MCX-ExEm two-pass
%  excitation/emission method (QUEL Imaging):
%  Nguyen MH, LaRochelle EPM, Robledo EA, Ruiz AJ, "Toward fluorescence
%  digital twins: multi-parameter experimental validation of fluorescence
%  Monte Carlo simulations using solid phantoms", J. Biomed. Opt. 30(S3),
%  S34104 (2025). doi:10.1117/1.JBO.30.S3.S34104
%  Reference configs: github.com/QUEL-Imaging/MCX-ExEm, under
%  "JSON Input Files/2.4.1 Cuvette" (cuvette_<conc>nM_step1/step2.json);
%  reference results: "Results Data.xlsx", sheet "Cuvette".
%
%  Geometry/optics below are taken from those JSON files: 10x10x20mm box,
%  mua_background=0.002/mm, mus=0, g=0, n=1.33, QY=1, and a 5mm-diameter
%  collimated disk beam entering the +x face and crossing the 10mm width.
%  MCX-ExEm's Pass 2 uses a voxel 'pattern3d' source with srcdir(4)=nan for
%  isotropic re-emission; this demo uses the mesh-native equivalent,
%  cfg.srctype='elembary' (mmclab.m), which shares that same nan-focus
%  convention and is also used by
%  demo_mcxyz_skinvessel_shallowvessel.m's vessel fluorescence estimate.
%
%  Part 1 sweeps the paper's six ICG concentrations and compares the
%  normalized emission-vs-concentration curve against MCX-ExEm's published
%  values; Part 2 plots the emission fluence map for a single concentration.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfg2 flux fluxem;

%% ICG optical properties
% NOTE: the paper's text quotes "~17,500" at 785nm, but its own
% cuvette_300nM_step1.json encodes mua=0.014263475/mm, which decomposes
% exactly as 0.002 + 2.303*17750*300e-9. The JSON value is used here.
% (The 820nm side needs no such correction: 0.002 + 2.303*8500*300e-9 =
% 0.00787265 matches cuvette_300nM_step2.json exactly.)
epsilonEx = 17750; % M^-1 mm^-1 @ 785nm, back-derived from the reference JSON
epsilonEm = 8500;  % M^-1 mm^-1 @ 820nm
QY = 1;
muaBackground = 0.002; % 1/mm

% the six concentrations of the paper's cuvette experiment, and MCX-ExEm's
% own simulated fluorescence intensities for them ("Results Data.xlsx",
% sheet "Cuvette", column "MCX-ExEm Simulation FL Intensity")
concList = [30 100 300 1000 3000 10000]; % nM
refSim = [0.0967027 0.3146472 0.88185966 2.3347218 3.8382010 2.5169575];
detailConc = 300; % the concentration whose fluence map is plotted in Part 2

%% build the 10x10x20mm cuvette mesh
% NOTE: the mesh must be built from a surface via surf2mesh rather than as a
% structured genT6mesh grid. mmcaddsrc() below retessellates the domain to
% embed the widefield source; on a structured T6 grid that retessellation
% aborts inside tetgen with "Assertion `ncollinear' failed" in
% scoutrefpoint(), because the grid's extracted surface is riddled with
% collinear/coplanar features its boundary recovery cannot resolve. An
% unstructured PLC from latticegrid+surf2mesh retessellates cleanly.

maxvol = 0.05; % mm^3, ~76k elements -- comparable to the old 0.5mm T6 grid
[nbox, fbox, c0] = latticegrid([0 10], [0 10], [0 20]);
c0(:, 4) = maxvol;
[cfg.node, cfg.elem] = surf2mesh(nbox, fbox, [], [], 1, [], c0);
cfg.elemprop = cfg.elem(:, 5);
cfg.elem = cfg.elem(:, 1:4);
cfg.unitinmm = 1;
cfg.method = 'elem';

%% excitation source: 5mm-diameter collimated disk beam (the paper's source)
% the reference JSON puts the source at Pos [100,50,100] voxels x 0.1mm =
% (10,5,10)mm on the +x face with Dir [-1 0 0], i.e. the beam crosses the
% 10mm width, NOT the 20mm length; Param1 [25] voxels = 2.5mm radius.

cfg.srcpos = [11 5 10];  % 1mm outside the +x face, so mmcaddsrc can enclose it
cfg.srcdir = [-1 0 0];
cfg.srctype = 'disk';
cfg.srcparam1 = [2.5 0 0 0]; % srcparam1(1) is the disk RADIUS -> 5mm diameter

%% extend the mesh to enclose the widefield source and retessellate

srcdef = struct('srctype', cfg.srctype, 'srcpos', cfg.srcpos, 'srcdir', cfg.srcdir, ...
                'srcparam1', cfg.srcparam1, 'srcparam2', []);

[cfg.node, cfg.elem] = mmcaddsrc(cfg.node, [cfg.elem cfg.elemprop], ...
                                 mmcsrcdomain(srcdef, [min(cfg.node); max(cfg.node)]));

cfg.elemprop = cfg.elem(:, 5);
cfg.elem = cfg.elem(:, 1:4);

[cfg.elem, cfg.evol] = meshreorient(cfg.node, cfg.elem(:, 1:4));
cfg.isreoriented = 1;
cfg.evol = abs(cfg.evol); % meshreorient() leaves evol signed; 'flux' needs true volumes

cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.outputtype = 'energy';
cfg.basisorder = 0;
cfg.nphoton = 1e6;
cfg.seed = 1648335518;
cfg.debuglevel = 'P';
cfg.isreflect = 1;  % reference JSON: "DoMismatch": true
cfg.isspecular = 1; % reference JSON: "DoSpecular": true

cuvetteidx = find(cfg.elemprop == 1);

%% Part 1: sweep the paper's concentrations, two passes each

signal = zeros(size(concList));
budget = zeros(size(concList));
escape = zeros(size(concList));

for ic = 1:numel(concList)
    conc = concList(ic) * 1e-9; % mol/L
    muaIcgEx = 2.303 * epsilonEx * conc;
    muaIcgEm = 2.303 * epsilonEm * conc;
    muaTotalEx = muaBackground + muaIcgEx;
    pFluor = (muaIcgEx / muaTotalEx) * QY;

    % ---- Pass 1: excitation (785nm) ----
    cfg.prop = [0 0 1 1          % 0: background/ambient
                muaTotalEx 0 0 1.33]; % 1: ICG cuvette medium at 785nm
    flux = mmclab(cfg);
    absorbedfrac = flux.data;

    % ---- fluorophore conversion: absorbed excitation -> emission weight ----
    w = absorbedfrac(cuvetteidx) * pFluor;
    mask = w > 0;
    srcidx = cuvetteidx(mask);
    w = w(mask);
    budget(ic) = sum(w);

    % ---- Pass 2: emission (820nm) from a weighted-element volumetric source ----
    cfg2 = cfg;
    cfg2.srctype = 'elembary';
    cfg2.srcparam1 = [numel(srcidx) 0 0 0];
    cfg2.srcpattern = reshape([cumsum(w)'; srcidx'], [], 1);
    cfg2.srcdir = [cfg.srcdir nan]; % nan focal length -> isotropic emission
    cfg2.e0 = 1; % skip tsearchn/widefield-source setup, unused by elembary
    cfg2.prop = [0 0 1 1                    % 0: background/ambient
                 muaBackground+muaIcgEm 0 0 1.33]; % 1: cuvette at 820nm (real re-absorption)
    fluxem = mmclab(cfg2);

    % elembary launches every photon at weight 1, so Pass 2 is normalized to
    % unit EMITTED energy. The detected signal is the emitted-energy budget
    % sum(w) times the fraction of that energy escaping re-absorption; the
    % drop in escape at high concentration is what produces the quenching
    % turnover the paper reports.
    escape(ic) = 1 - sum(fluxem.data);
    signal(ic) = budget(ic) * escape(ic);

    fprintf(1, ['C=%6g nM | mua_ex=%.6f mua_em=%.6f pFluor=%.4f | ' ...
                'srcelem=%d sum(w)=%.5f escape=%.4f signal=%.6g\n'], ...
            concList(ic), muaTotalEx, muaBackground + muaIcgEm, pFluor, ...
            numel(srcidx), budget(ic), escape(ic), signal(ic));

    % keep the detail-concentration setup for the Part 2 fluence map
    if (concList(ic) == detailConc)
        detailcfg = cfg2;
        detailw = w;
    end
end

%% compare the normalized curve against MCX-ExEm's published values

[~, ipk] = max(refSim);
ours = signal / signal(ipk);
theirs = refSim / refSim(ipk);

fprintf(1, '\n=== normalized emission vs concentration (each scaled to its own peak) ===\n');
fprintf(1, '  conc(nM)   this demo    MCX-ExEm       ratio\n');
for ic = 1:numel(concList)
    fprintf(1, '  %8g   %9.4f   %9.4f   %9.3f\n', ...
            concList(ic), ours(ic), theirs(ic), ours(ic) / theirs(ic));
end
fprintf(1, '  max normalized absolute error: %.4f\n', max(abs(ours - theirs)));
fprintf(1, '  peak: this demo %g nM, MCX-ExEm %g nM\n', ...
        concList(signal == max(signal)), concList(ipk));

figure;
subplot(121);
semilogx(concList, ours, 'o-', concList, theirs, 's--', 'linewidth', 1.5);
xlabel('ICG concentration (nM)');
ylabel('normalized fluorescence emission');
legend('this demo (MMC, elembary)', 'MCX-ExEm (published)', 'location', 'northwest');
title('emission vs concentration');
grid on;
box on;

%% Part 2: emission fluence map at the detail concentration

cfg2 = detailcfg;
cfg2.outputtype = 'flux';
cfg2.basisorder = 1;
fluxem = mmclab(cfg2);
fluxemcw = sum(fluxem.data, 2) * cfg2.tstep * sum(detailw);

subplot(122);
% cut through the beam axis (the beam runs along -x at y=5, z=10)
qmeshcut(cfg2.elem(cfg2.elemprop > 0, 1:4), cfg2.node, log10(fluxemcw), 'y=5', 'linestyle', 'none');
view([0 1 0]);
box on;
axis equal;
title(sprintf('820nm emission fluence, %g nM ICG', detailConc));
colorbar;
colormap(jet);
