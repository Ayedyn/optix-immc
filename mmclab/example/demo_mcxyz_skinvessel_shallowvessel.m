%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  mcxyz skinvessel benchmark - shallow vessel variant, with a two-pass
%  excitation/emission fluorescence estimate in the vessel
%
%  modified from demo_mcxyz_skinvessel.m:
%    - epidermis thinned to a 0.05 mm surface layer (starting at the skin
%      surface, i.e. no vacuum gap above the tissue as in the original)
%    - vessel cylinder kept at 0.2 mm diameter
%    - vessel is repositioned to sit ~0.2 mm deep into the dermis (measured
%      from the epidermis/dermis boundary), instead of the ~0.35 mm depth
%      used in the original benchmark
%
%  must change mcxyz maketissue.m boundaryflag variable from 2 to 1 to get
%  comparable absorption fraction (40%), otherwise, mcxyz obtains slightly
%  higher absorption (~42%) with boundaryflag=2
%
%  Part 2 (added) estimates fluorescence emitted from the vessel: a fixed
%  fraction of the vessel's excitation-wavelength mua is treated as a
%  fluorophore, converted to a re-emission weight per vessel element, and
%  propagated with cfg.srctype='elembary' (weighted-element + barycentric
%  volumetric source, mmclab.m) at the emission wavelength's properties.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg flux;

%% create the skin-vessel benchmark mesh
% z-boundaries below are in grid units, where 1 unit = cfg.unitinmm = 0.005 mm
%   0   - skin surface (top of the tissue)
%   10  - epidermis/dermis boundary (epidermis thickness = 10 units = 0.05 mm)
%   200 - bottom of the domain
[no, fc] = latticegrid([0 200], [0 200], [0 10 200]); % epidermis + dermis layers
no(end, :) = no(end, :) + 1e-5;

fc2 = cell2mat(fc);
fc = [fc2(:, [1 2 3]); fc2(:, [1 3 4])];

% vessel: radius = 20 grid units (0.1 mm) -> 0.2 mm diameter, centered at
% z = 50.5, i.e. 40.5 grid units (~0.2 mm) below the epidermis/dermis
% boundary at z=10, so the vessel sits ~0.2 mm deep in the dermis.
% NOTE: tsize=8 (coarser than the 5 used in demo_mcxyz_skinvessel.m) is
% required here - at this shallower depth, tsize=5 triggers a tetgen
% boundary-recovery crash both in the region-tagged tetrahedralization
% below and in mmcaddsrc's mesh-refinement step further down; tsize=8
% avoids both while keeping the same cylinder radius/geometry.
[ncy, fcy] = meshacylinder([-1, 99.5, 50.5], [201, 99.5, 50.5], 20, 8); % add the vessel
[newnode, newelem] = surfboolean(no, fc, 'first', ncy, fcy);  % merge the two domains

c0 = [5, 150, 50]';   % seed points (z-coordinate) for: epidermis, dermis, vessel
seeds = [ones(3, 2) * 100, c0];  % define the regions by index

% ISO2MESH_TETGENOPT='-Y -A'
[cfg.node, cfg.elem] = s2m(newnode, newelem(:, 1:3), 1, 30, 'tetgen', seeds, []); % creating the merged mesh domain

cfg.unitinmm = 0.005;
cfg.method = 'elem';

figure;
subplot(131);
plotmesh(cfg.node, cfg.elem);

cfg.elemprop = cfg.elem(:, 5);
cfg.elem = cfg.elem(:, 1:4);

%% define other properties (excitation wavelength)
%             mua        mus         g       n
cfg.prop = [0.0000     0.0000    1.0000    1         % 0: background
            1.6572    37.5940    0.9000    1.3700    % 1: epidermis
            0.0458    35.6541    0.9000    1.3700    % 2: dermis
            23.0543     9.3985    0.9000    1.3700]; % 3: vessel (blood)

cfg.srcpos = [100 100 -1];
cfg.srcdir = [0 0 1];

cfg.tstart = 0;
cfg.tend = 5e-8;
cfg.tstep = 5e-8;
% per-element absorbed energy (fraction of cfg.nphoton), needed as-is for
% the Part 2 fluorescence weight calc -- no mua/evol reconstruction needed
cfg.outputtype = 'energy';
cfg.basisorder = 0;
cfg.minenergy = 0.01;

cfg.srctype = 'disk';
cfg.srcparam1 = [0.3 0 0 0] / cfg.unitinmm; % in grid unit

%% define wide-field disk source by extending the mesh to the widefield src
srcdef = struct('srctype', cfg.srctype, 'srcpos', cfg.srcpos, 'srcdir', cfg.srcdir, ...
                'srcparam1', cfg.srcparam1, 'srcparam2', []);

[cfg.node, cfg.elem] = mmcaddsrc(cfg.node, [cfg.elem cfg.elemprop], ...
                                 mmcsrcdomain(srcdef, [min(cfg.node); max(cfg.node)]));

cfg.elemprop = cfg.elem(:, 5);
cfg.elem = cfg.elem(:, 1:4);

% At this shallow vessel depth, tetgen occasionally emits a handful of
% exactly-zero-volume (degenerate) tetrahedra where the vessel surface
% meets the box's x=0/x=200 faces; which elements this affects is
% tetgen-version/platform dependent. A zero-volume element carries no
% photon weight, so it is safe to simply drop it from the mesh rather
% than fight tetgen's numerics for a specific parameter combination.
[cfg.elem, cfg.evol] = meshreorient(cfg.node, cfg.elem(:, 1:4));
degenidx = find(cfg.evol == 0);
if (~isempty(degenidx))
    fprintf(1, 'removing %d degenerate (zero-volume) element(s): [%s]\n', ...
            numel(degenidx), sprintf('%d ', degenidx));
    cfg.elem(degenidx, :) = [];
    cfg.elemprop(degenidx) = [];
    cfg.evol(degenidx) = [];
end
cfg.isreoriented = 1;

%% other simulation information

cfg.nphoton = 1e7;
cfg.seed = 1648335518;

cfg.debuglevel = 'TP';
cfg.isreflect = 0;

%% Pass 1: excitation forward simulation

flux = mmclab(cfg);
absorbedfrac = flux.data; % per-element, fraction of cfg.nphoton absorbed there

%% plot Pass 1 excitation absorption (nodal-averaged from per-element data for qmeshcut)

nodeval = accumarray(cfg.elem(:), repmat(absorbedfrac, 4, 1)) ./ accumarray(cfg.elem(:), 1);

subplot(132);
hold on;
qmeshcut(cfg.elem(cfg.elemprop > 0, 1:4), cfg.node * cfg.unitinmm, log10(nodeval), 'x=0.5', 'linestyle', 'none');
view([1 0 0]);
set(gca, 'zlim', [0 1], 'ylim', [0 1], 'zdir', 'reverse');
box on;
axis equal;
title('Pass 1: excitation absorbed fraction per element');
colorbar;
colormap(jet);

%% Part 2: two-pass fluorescence estimate in the vessel
% fluorFraction of the vessel's excitation mua is treated as a fluorophore;
% QY is its quantum yield -- both are demo choices, not literature values.
fluorFraction = 0.5;
QY = 0.1;
pFluor = fluorFraction * QY;

vesselidx = find(cfg.elemprop == 3);
w = absorbedfrac(vesselidx) * pFluor;
mask = w > 0;
srcidx = vesselidx(mask);
w = w(mask);

fprintf(1, 'fluorescence source: %d of %d vessel elements have nonzero weight\n', ...
        numel(srcidx), numel(vesselidx));
fprintf(1, 'total emission weight budget (fraction of Pass 1 launched weight): %g\n', sum(w));

%% Pass 2: emission simulation using the new elembary volumetric source

cfg2 = cfg;
cfg2.evol = abs(cfg.evol); % meshreorient() leaves evol signed; 'flux' output needs true volumes
cfg2.srctype = 'elembary';
cfg2.srcparam1 = [numel(srcidx) 0 0 0];
cfg2.srcpattern = reshape([cumsum(w)'; srcidx'], [], 1);
cfg2.srcdir = [0 0 1 nan]; % isotropic emission
cfg2.e0 = 1; % skip tsearchn/widefield-source setup, unused by elembary
% emission-wavelength optical properties: reuse the excitation table, but
% drop the fluorophore-attributable share of the vessel's mua so a photon
% re-absorbed on its way out isn't double counted as further fluorescence
cfg2.prop(4, 1) = cfg.prop(4, 1) * (1 - fluorFraction);
cfg2.outputtype = 'flux';
cfg2.basisorder = 1;

fluxem = mmclab(cfg2);
fluxemcw = sum(fluxem.data, 2) * cfg2.tstep * 100;

%% plot Pass 2 fluorescence emission fluence

subplot(133);
hold on;
qmeshcut(cfg2.elem(cfg2.elemprop > 0, 1:4), cfg2.node * cfg2.unitinmm, log10(fluxemcw), 'x=0.5', 'linestyle', 'none');
view([1 0 0]);
set(gca, 'zlim', [0 1], 'ylim', [0 1], 'zdir', 'reverse');
box on;
axis equal;
title('Pass 2: estimated fluorescence emission fluence (W/cm^2 per W)');
colorbar;
colormap(jet);
