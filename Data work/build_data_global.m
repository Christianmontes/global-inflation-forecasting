clear
close all
clc

% =========================================================================
% DESCRIPTION
% Builds the processed datasets from the raw CPI-index workbooks. For each
% sample (original 1980-2019 and extended 1980-2023) the script:
%
%   1) transforms every series to monthly inflation using its
%      transformation code (tcode 5 = first difference of natural log)
%      via prepare_missing(), and drops the first month, which is lost
%      to differencing;
%
%   2) fills observations that are missing in the raw data with the
%      EM algorithm of factors_em() (kmax=99 -> 8 factors, jj=2,
%      DEMEAN=2); only the EM-completed dataset (output x2) is kept,
%      the factor estimates themselves are discarded;
%
%   3) adds the continent header row from the matching continent list
%      and writes the result to ../Processed data/.
%
% No outlier removal is performed, and no factors are estimated beyond
% what the EM imputation itself requires.
%
% INPUT  (../Data/)
%   Raw workbooks, layout: row 1 = country names, row 2 = transformation
%   codes, column A = dates as text (M/d/yyyy), data from row 3.
%   Continent lists, layout: row 1 = continent labels, one column per
%   country, in the same column order as the matching raw workbook.
%
% OUTPUT (../Processed data/)
%   One workbook per sample, layout: row 1 = continent, row 2 = country,
%   column A = dates, columns in raw-file order.
%
% AUXILIARY FUNCTIONS (same folder as this script)
%   prepare_missing.m - transforms series based on tcodes
%   factors_em.m      - EM algorithm used to fill missing observations
%
% -------------------------------------------------------------------------
% PARAMETERS TO BE CHANGED

% Datasets to build: {raw workbook, continent list, output name}
datasets = { ...
    'Raw data.xlsx',          'Continentlist.xlsx',          'Data_Global.xlsx'; ...
    'Raw data extended.xlsx', 'Continentlist extended.xlsx', 'Data_Global_extended.xlsx'};

% factors_em() settings: kmax=99 forces 8 factors; jj = information
% criterion (unused when kmax=99); DEMEAN=2 demeans and standardizes
% before the PCA step
kmax   = 99;
jj     = 2;
DEMEAN = 2;

% =========================================================================
% BUILD EACH DATASET

% Resolve paths relative to this script so it runs from any directory
script_dir = fileparts(mfilename('fullpath'));

for k = 1:size(datasets, 1)
    raw_file  = fullfile(script_dir, '..', 'Data', datasets{k, 1});
    cont_file = fullfile(script_dir, '..', 'Data', datasets{k, 2});
    out_file  = fullfile(script_dir, '..', 'Processed data', datasets{k, 3});

    % ---------------------------------------------------------------------
    % PART 1: LOAD AND LABEL DATA

    C = readcell(raw_file);

    % Country names (row 1) and transformation codes (row 2)
    series = string(C(1, 2:end));
    tcode  = cell2mat(C(2, 2:end));

    % Dates (column A, text of the form M/d/yyyy)
    dates = datetime(string(C(3:end, 1)), 'InputFormat', 'M/d/yyyy');
    T     = numel(dates);
    N     = numel(series);

    % Numeric block; empty cells (missing observations) become NaN
    vals           = C(3:end, 2:end);
    isnum          = cellfun(@(v) isnumeric(v) && ~isempty(v), vals);
    rawdata        = NaN(T, N);
    rawdata(isnum) = cell2mat(vals(isnum));

    % Continent labels (row 1 of the continent list, same column order as
    % the raw workbook)
    continent = string(readcell(cont_file));
    continent = continent(1, :);
    assert(numel(continent) == N, ...
        '%s has %d entries but %s has %d countries.', ...
        datasets{k, 2}, numel(continent), datasets{k, 1}, N);

    % ---------------------------------------------------------------------
    % PART 2: PROCESS DATA

    % Transform raw data to be stationary using auxiliary function
    % prepare_missing()
    yt = prepare_missing(rawdata, tcode);

    % Reduce sample to usable dates: remove first month because the
    % series have been first differenced
    yt    = yt(2:T, :);
    dates = dates(2:T);

    % Fill missing observations with the EM algorithm of factors_em();
    % x2 is the dataset with missing values replaced, all other outputs
    % (factor estimates) are discarded
    [~, ~, ~, ~, x2] = factors_em(yt, kmax, jj, DEMEAN);

    % ---------------------------------------------------------------------
    % PART 3: WRITE OUTPUT

    out               = cell(T - 1 + 2, N + 1);
    out(1, :)         = [{'Continent'}, cellstr(continent)];
    out(2, :)         = [{'Date'},      cellstr(series)];
    out(3:end, 1)     = cellstr(string(dates, 'M/d/uuuu'));
    out(3:end, 2:end) = num2cell(x2);

    writecell(out, out_file, 'WriteMode', 'replacefile');
    fprintf('%s: %d months x %d countries, %d missing cells filled by EM\n', ...
        datasets{k, 3}, T - 1, N, nnz(isnan(yt)));
end
