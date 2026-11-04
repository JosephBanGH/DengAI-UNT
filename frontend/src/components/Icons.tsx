import type { SVGProps } from "react";

type IconName =
  | "dashboard"
  | "data"
  | "dataset"
  | "ml"
  | "forecast"
  | "map"
  | "alert"
  | "report"
  | "assistant"
  | "workflow"
  | "audit"
  | "users"
  | "roles"
  | "settings"
  | "logout"
  | "search"
  | "download"
  | "plus"
  | "check"
  | "close"
  | "play"
  | "refresh"
  | "chevron"
  | "info";

const PATHS: Record<IconName, string> = {
  dashboard: "M4 13h6V4H4v9Zm0 7h6v-5H4v5Zm9 0h7V11h-7v9Zm0-16v5h7V4h-7Z",
  data: "M12 3c-4.4 0-8 1.3-8 3s3.6 3 8 3 8-1.3 8-3-3.6-3-8-3Zm8 8c0 1.7-3.6 3-8 3s-8-1.3-8-3V8.2C5.6 9.3 8.6 10 12 10s6.4-.7 8-1.8V11Zm0 6c0 1.7-3.6 3-8 3s-8-1.3-8-3v-2.8c1.6 1.1 4.6 1.8 8 1.8s6.4-.7 8-1.8V17Z",
  dataset: "M4 5h16v4H4V5Zm0 6h16v4H4v-4Zm0 6h10v3H4v-3Z",
  ml: "M12 2 2 7l10 5 10-5-10-5Zm0 8.5L2 15l10 5 10-5-10-4.5Z",
  forecast: "M3 17h2l3-6 3 4 3-7 3 5h4",
  map: "M9 2 3 5v17l6-3 6 3 6-3V2l-6 3-6-3Zm0 2.2 4 2v13.6l-4-2V4.2Zm6 2 4-2v13.6l-4 2V6.2Z",
  alert: "M12 2 1 21h22L12 2Zm1 14h-2v-2h2v2Zm0-4h-2V7h2v5Z",
  report: "M6 2h8l4 4v16H6V2Zm7 1.5V7h3.5L13 3.5ZM8 11h8v2H8v-2Zm0 4h8v2H8v-2Z",
  assistant: "M12 3a7 7 0 0 1 7 7v1a5 5 0 0 1-5 5h-1v3h-2v-3H9a5 5 0 0 1-5-5v-1a7 7 0 0 1 7-7Zm-2.5 6a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm5 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z",
  workflow: "M4 4h5v5H4V4Zm11 11h5v5h-5v-5ZM4 15h5v5H4v-5Zm6.5-8.5H15V12h1.5V4.5h-6v4h1.5V6.5h-1.5Z",
  audit: "M5 3h9l5 5v13H5V3Zm8 1.5V9h3.5L13 4.5ZM8 13h8v2H8v-2Zm0 4h5v2H8v-2Z",
  users: "M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm7 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM2 20c0-3.3 3.1-6 7-6s7 2.7 7 6H2Zm14.5-5.5c2.6.3 4.5 2.6 4.5 5.5h-5.2c0-2.2-.8-4-2.3-5.2.9-.2 1.9-.3 3-.3Z",
  roles: "M12 2 4 6v6c0 5 3.4 9.4 8 10 4.6-.6 8-5 8-10V6l-8-4Zm0 2.2 6 3V12c0 4-2.6 7.6-6 8.1-3.4-.5-6-4.1-6-8.1V7.2l6-3Z",
  settings: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm9 4-2.1-.6a7 7 0 0 0-.7-1.7l1.1-1.9-2-2-1.9 1.1a7 7 0 0 0-1.7-.7L12.7 4h-2.8l-.6 2.1a7 7 0 0 0-1.7.7L5.7 5.7l-2 2 1.1 1.9a7 7 0 0 0-.7 1.7L2 12v2.8l2.1.6a7 7 0 0 0 .7 1.7l-1.1 1.9 2 2 1.9-1.1a7 7 0 0 0 1.7.7l.6 2.1h2.8l.6-2.1a7 7 0 0 0 1.7-.7l1.9 1.1 2-2-1.1-1.9a7 7 0 0 0 .7-1.7L22 14v-2Z",
  logout: "M10 3H5v18h5v-2H7V5h3V3Zm4.6 4.6L13.2 9l2 2H9v2h6.2l-2 2 1.4 1.4L19 12l-4.4-4.4Z",
  search: "M10 2a8 8 0 1 0 4.9 14.3l5.4 5.4 1.4-1.4-5.4-5.4A8 8 0 0 0 10 2Zm0 2a6 6 0 1 1 0 12 6 6 0 0 1 0-12Z",
  download: "M11 3h2v9h3.5L12 17.5 7.5 12H11V3ZM4 19h16v2H4v-2Z",
  plus: "M11 4h2v7h7v2h-7v7h-2v-7H4v-2h7V4Z",
  check: "m9 16.2-3.5-3.5L4 14.2 9 19 20 8l-1.4-1.4L9 16.2Z",
  close: "M18.3 5.7 12 12l6.3 6.3-1.4 1.4L10.6 13.4 4.3 19.7 2.9 18.3 9.2 12 2.9 5.7 4.3 4.3 10.6 10.6 16.9 4.3l1.4 1.4Z",
  play: "M8 5v14l11-7L8 5Z",
  refresh: "M12 4V1L8 5l4 4V6a6 6 0 1 1-6 6H4a8 8 0 1 0 8-8Z",
  chevron: "m9 6 6 6-6 6",
  info: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm1 15h-2v-6h2v6Zm0-8h-2V7h2v2Z",
};

interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
  size?: number;
}

export function Icon({ name, size = 16, ...rest }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="currentColor"
      aria-hidden="true"
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}

export type { IconName };
