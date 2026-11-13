export interface RoleSummary {
  slug: string;
  name: string;
}

export interface Me {
  id: number;
  email: string;
  full_name: string;
  position: string;
  organization: string;
  roles: RoleSummary[];
  permissions: string[];
  is_superuser: boolean;
  must_change_password: boolean;
  last_login_at: string | null;
}

export interface Kpi {
  key: string;
  label: string;
  value: number;
  unit?: string;
  change_pct?: number | null;
  trend?: string | null;
  hint?: string | null;
}

export interface DashboardOverview {
  generated_at: string;
  data_freshness: string | null;
  is_synthetic_data: boolean;
  kpis: Kpi[];
  weekly_series: { week: string; cases: number; label: string }[];
  districts_top_incidence: {
    district_code: string;
    district_name: string;
    province: string;
    cases_12w: number;
    incidence_per_100k: number | null;
    population: number | null;
  }[];
  districts_top_risk: {
    district_code: string;
    district_name: string;
    level: string;
    score: number;
    predicted_cases: number | null;
    incidence_per_100k: number | null;
  }[];
  risk_distribution: { level: string; count: number }[];
  active_alerts: number;
  alerts_by_level: { level: string; count: number }[];
  climate_summary: { variable: string; value: number | null; unit: string; change_pct: number | null }[];
  model_performance: {
    algorithm: string;
    version: string;
    horizon: number | null;
    MAE: number | null;
    RMSE: number | null;
    AUC_PR: number | null;
    coverage: number | null;
    promoted_at: string | null;
  }[];
  integration_status: { code: string; name: string; kind: string; status: string; is_synthetic: boolean; last_success_at: string | null }[];
  data_quality: Record<string, unknown>;
  operations?: Record<string, number>;
}

export interface GeoFeature {
  type: "Feature";
  id?: string;
  geometry: { type: string; coordinates: number[][][] | number[][][][] } | null;
  properties: Record<string, any>;
}

export interface FeatureCollection {
  type: "FeatureCollection";
  features: GeoFeature[];
  metadata?: {
    thresholds?: { low: number; moderate: number; high: number; method?: string; justification?: string };
    generated_at?: string | null;
    legend?: { level: string; label: string; min: number; max: number | null }[];
    window_weeks?: number;
  };
}

export interface Prediction {
  district_code: string;
  district_name: string;
  target_week_start: string;
  target_year: number;
  target_week: number;
  horizon_weeks: number;
  predicted_cases: number;
  lower_bound: number | null;
  upper_bound: number | null;
  risk_level: string;
  risk_score: number | null;
  incidence_per_100k: number | null;
}

export interface Algorithm {
  key: string;
  name: string;
  family: string;
  role: string;
  description: string;
  spatial: boolean;
  temporal: boolean;
  attention: boolean;
  backend: string;
  status: string;
  status_reason?: string;
  default_horizons: number[];
}

export interface Experiment {
  id: number;
  run_id: string;
  name: string;
  algorithm: string;
  status: string;
  stage: string;
  progress: number;
  n_trials: number;
  trials_completed: number;
  validation_strategy: string;
  horizons: { values?: number[] } | number[] | null;
  hyperparameters: Record<string, Record<string, number>> | null;
  metrics: Record<string, { metrics: Record<string, number | null>; best_params: Record<string, number>; n_train: number; n_val: number; n_test: number; notes: Record<string, unknown> }> | null;
  best_trial: Record<string, unknown> | null;
  mlflow_run_id: string;
  is_synthetic_data: boolean;
  notes: string;
  error_message: string;
  started_at: string;
  finished_at: string | null;
  duration_seconds: number | null;
  created_by: string;
  running?: boolean;
}

export interface ModelVersion {
  id: number;
  algorithm: string;
  version: string;
  status: string;
  is_production: boolean;
  training_weeks: number;
  metrics: Record<string, number | null> | null;
  hyperparameters: Record<string, number> | null;
  promoted_at: string | null;
  promoted_by: string;
  created_at: string;
  artifact_checksum: string;
}

export interface AlertItem {
  id: number;
  alert_key: string;
  district_code: string;
  district_name: string;
  epi_year: number;
  epi_week: number;
  horizon_weeks: number;
  risk_level: string;
  risk_score: number | null;
  predicted_cases: number | null;
  lower_bound: number | null;
  upper_bound: number | null;
  incidence_per_100k: number | null;
  rationale: string;
  status: string;
  acknowledged_by: string;
  acknowledged_at: string | null;
  closed_by: string;
  closed_at: string | null;
  notes: string;
  generated_at: string;
  is_synthetic_data: boolean;
}

export interface ReportItem {
  id: number;
  report_key: string;
  title: string;
  category: string;
  report_type: string;
  format: string;
  status: string;
  file_size: number;
  filters: Record<string, unknown> | null;
  period_start: string | null;
  period_end: string | null;
  model_version: string;
  generated_by: string;
  generated_at: string;
  is_synthetic_data: boolean;
  methodology_notes: string;
  error_message: string;
}

export interface ReportType {
  key: string;
  category: string;
  title: string;
  description: string;
  formats: string[];
  requires_period: boolean;
}

export interface AuditEvent {
  id: number;
  occurred_at: string;
  event_type: string;
  module: string;
  action: string;
  result: string;
  user_email: string;
  user_role: string;
  session_id: string;
  resource_type: string;
  resource_id: string;
  description: string;
  ip_address: string;
  duration_ms: number | null;
  hash: string;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

export interface UserItem {
  id: number;
  email: string;
  full_name: string;
  position: string;
  organization: string;
  is_active: boolean;
  is_superuser: boolean;
  must_change_password: boolean;
  last_login_at: string | null;
  created_at: string;
  locked_until: string | null;
  roles: { id: number; slug: string; name: string }[];
}

export interface RoleItem {
  id: number;
  slug: string;
  name: string;
  description: string;
  is_system: boolean;
  user_count: number;
  permissions: { code: string; description: string; module: string }[];
}

export interface PermissionItem {
  code: string;
  description: string;
  module: string;
}

export interface WorkflowItem {
  code: string;
  name: string;
  description: string;
  schedule: string;
  file: string;
  steps: string[];
  last_execution: {
    id: number;
    status: string;
    started_at: string;
    duration_ms: number | null;
    items_processed: number;
  } | null;
}

export interface ExplanationFactor {
  feature: string;
  label: string;
  contribution: number;
  direction: string;
  value: number | null;
}

export interface Explanation {
  district_code: string;
  district_name: string;
  model_version: string;
  algorithm: string;
  method: string;
  base_value: number | null;
  prediction: number | null;
  factors: ExplanationFactor[];
  temporal_importance: { step?: number; lag?: number; delta?: number; abs_contribution?: number }[];
  narrative: string;
  caveat: string;
}

export interface AssistantAnswer {
  answer: string;
  grounded: boolean;
  provider: string;
  sources: { document: string; section: string; score: number; snippet: string }[];
  facts: Record<string, unknown>;
  caveat: string;
  latency_ms: number | null;
}

export interface DatasetItem {
  id: number;
  slug: string;
  name: string;
  version: string;
  description: string;
  rows: number;
  districts: number;
  weeks: number;
  start_week: string | null;
  end_week: string | null;
  is_synthetic: boolean;
  status: string;
  checksum: string;
  created_at: string;
  created_by: string;
  source_summary: Record<string, unknown> | null;
  feature_schema: Record<string, unknown> | null;
}

export interface DataDictionaryItem {
  column: string;
  dtype: string;
  description: string;
  unit: string;
  source: string;
  nullable: boolean;
}

export interface GroupValidationCheck {
  id: string;
  title: string;
  description: string;
  status: "PASSED" | "FAILED" | "WARNING";
  details: string;
}

export interface GroupValidationResult {
  audited_at: string;
  dataset_id: number | null;
  dataset_name: string;
  total_records: number;
  is_synthetic: boolean;
  data_mode: string;
  groups_summary: {
    districts_count: number;
    weeks_count: number;
    start_week: string;
    end_week: string;
    balance_ratio: number;
    districts_detail: { code: string; name: string; province: string; records: number; pct: number }[];
  };
  partitions_audit: {
    temporal_walk_forward: {
      train_rows: number;
      val_rows: number;
      test_rows: number;
      train_period: string;
      val_period: string;
      test_period: string;
      overlap_weeks: number;
      status: string;
    };
    spatial_block_cv: {
      train_districts: string[];
      val_districts: string[];
      test_districts: string[];
      overlap_districts: number;
      status: string;
    };
  };
  leakage_risk: {
    risk_level: "LOW" | "MEDIUM" | "HIGH";
    risk_score_pct: number;
    total_checks: number;
    passed_checks: number;
    verifications: GroupValidationCheck[];
  };
  methodological_recommendations: string[];
}

export interface UnivariateStatistic {
  variable: string;
  unit: string;
  count: number;
  valid_count: number;
  missing_count: number;
  missing_pct: number;
  mean: number | null;
  std: number | null;
  median: number | null;
  min: number | null;
  q25: number | null;
  q75: number | null;
  iqr: number | null;
  max: number | null;
  skewness: number | null;
  kurtosis: number | null;
}

export interface BivariateCorrelation {
  feature_x: string;
  feature_y: string;
  pearson: number;
  pearson_p_value: number;
  spearman: number;
  spearman_p_value: number;
  significant_05: boolean;
}

export interface StatisticalExplanation {
  topic: string;
  indicator: string;
  behavior: string;
  methodological_implication: string;
  epistemic_caution: string;
}

export interface DatasetStatisticsResult {
  generated_at: string;
  dataset_id: number | null;
  dataset_name: string;
  total_records: number;
  is_synthetic: boolean;
  univariate_statistics: UnivariateStatistic[];
  correlations: BivariateCorrelation[];
  district_breakdown: {
    district_code: string;
    total_cases: number;
    mean_cases_week: number;
    max_cases_week: number;
    mean_temp: number | null;
    weeks_count: number;
  }[];
  automated_explanations: StatisticalExplanation[];
}


export interface DataSourceItem {
  id: number;
  code: string;
  name: string;
  kind: string;
  provider: string;
  url: string;
  license: string;
  access_method: string;
  integration_status: string;
  is_synthetic: boolean;
  refresh_frequency: string;
  last_success_at: string | null;
  last_error: string;
  notes: string;
}

export interface QualityCheck {
  id: number;
  dataset_name: string;
  check_name: string;
  severity: string;
  rows_checked: number;
  rows_flagged: number;
  details: Record<string, unknown> | null;
  created_at: string;
}

export interface QualitySummary {
  total_cases: number;
  total_climate: number;
  districts_covered: number;
  weeks_covered: number;
  missing_climate_pct: number;
  duplicate_rows: number;
  anomalies: number;
  synthetic_share_pct: number;
  checks: QualityCheck[];
}

export interface ImportJob {
  id: number;
  job_key: string;
  source_code: string | null;
  dataset_name: string;
  kind: string;
  status: string;
  file_name: string;
  rows_total: number;
  rows_inserted: number;
  rows_updated: number;
  rows_rejected: number;
  rows_duplicated: number;
  started_at: string;
  finished_at: string | null;
  duration_ms: number | null;
  initiated_by: string;
  is_synthetic: boolean;
  error_message: string;
}

export interface Health {
  status: string;
  version: string;
  environment: string;
  database: string;
  database_url_scheme: string;
  postgis: boolean;
  ml_backend: {
    torch: { available: boolean; version?: string };
    xgboost: { available: boolean; version?: string };
  };
  integrations: Record<string, string>;
  timestamp: string;
  limitations: string[];
}

export interface Comparison {
  rows: { algorithm: string; split: string; horizon_weeks: number; metric: string; value: number; ci_lower?: number | null; ci_upper?: number | null; n_observations: number }[];
  ranking: {
    algorithm: string;
    MAE: number;
    MAE_ci_lower?: number | null;
    MAE_ci_upper?: number | null;
    RMSE: number;
    RMSE_ci_lower?: number | null;
    RMSE_ci_upper?: number | null;
    MAPE?: number | null;
    MASE?: number | null;
    AUC_PR?: number | null;
    sensitivity?: number | null;
    specificity?: number | null;
    Brier?: number | null;
  }[];
  tests: Record<string, any>[];
  winner: string | null;
  winner_basis: string;
  horizon: number;
  n_observations: number;
  caveat: string;
  artifacts?: {
    filename: string;
    title: string;
    kind: string;
    format: string;
    description: string;
    available: boolean;
    size_bytes: number;
    download_url: string;
  }[];
  interpretations?: {
    id: string;
    title: string;
    summary: string;
    epidemiological_rationale: string;
  }[];
}

export interface ConfigEntry {
  key: string;
  value: any;
  description: string;
  updated_at: string | null;
  updated_by: string;
}
