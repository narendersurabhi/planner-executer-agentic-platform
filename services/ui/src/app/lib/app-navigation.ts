export type AppScreenId =
  | "home"
  | "project"
  | "compose"
  | "chat"
  | "workflows"
  | "studio"
  | "memory"
  | "rag";

export type AppNavIcon = "home" | "menu" | "palette" | "chat" | "library" | "graph" | "inspect";

export type AppNavItem = {
  id: AppScreenId;
  label: string;
  href: string;
  icon: AppNavIcon;
};

export const PRIMARY_APP_NAV_ITEMS: AppNavItem[] = [
  { id: "home", label: "Home", href: "/", icon: "home" },
  { id: "project", label: "Project", href: "/project", icon: "menu" },
  { id: "compose", label: "Request", href: "/compose", icon: "palette" },
  { id: "chat", label: "Workflow Chat", href: "/chat", icon: "chat" },
  { id: "workflows", label: "Saved Workflows", href: "/workflows", icon: "library" },
  { id: "studio", label: "Builder", href: "/studio", icon: "graph" },
  { id: "memory", label: "Context", href: "/memory", icon: "inspect" },
  { id: "rag", label: "Knowledge", href: "/rag", icon: "library" },
];
