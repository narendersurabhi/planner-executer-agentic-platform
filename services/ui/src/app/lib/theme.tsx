"use client";

import {
  createContext,
  startTransition,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type AppTheme = "dark" | "light";

const THEME_STORAGE_KEY = "ape.ui.theme.v1";

type AppThemeContextValue = {
  theme: AppTheme;
  mounted: boolean;
  setTheme: (nextTheme: AppTheme) => void;
  toggleTheme: () => void;
};

const AppThemeContext = createContext<AppThemeContextValue | null>(null);

const resolveInitialTheme = (): AppTheme => {
  if (typeof window === "undefined") {
    return "dark";
  }

  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === "dark" || storedTheme === "light") {
    return storedTheme;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
};

export function AppThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<AppTheme>("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const nextTheme = resolveInitialTheme();
    setThemeState(nextTheme);
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted || typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
  }, [mounted, theme]);

  const setTheme = (nextTheme: AppTheme) => {
    startTransition(() => {
      setThemeState(nextTheme);
    });
  };

  const value = useMemo<AppThemeContextValue>(
    () => ({
      theme,
      mounted,
      setTheme,
      toggleTheme: () => setTheme(theme === "dark" ? "light" : "dark"),
    }),
    [mounted, theme]
  );

  return <AppThemeContext.Provider value={value}>{children}</AppThemeContext.Provider>;
}

export const useAppTheme = () => {
  const value = useContext(AppThemeContext);
  if (!value) {
    throw new Error("useAppTheme must be used within AppThemeProvider.");
  }
  return value;
};
