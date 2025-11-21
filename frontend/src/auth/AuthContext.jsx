import React, { createContext, useContext, useState, useEffect } from "react";

const AuthContext = createContext();

const STORAGE_KEY = "lung_ai_user_v1";

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) setUser(JSON.parse(raw));
  }, []);

  function login({ email, password, role = "user" }) {
    // TODO: replace with actual backend auth
    const fakeUser = { id: Date.now().toString(), email, role, name: email.split("@")[0] };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(fakeUser));
    setUser(fakeUser);
    return fakeUser;
  }

  function signup({ name, email, password, role = "user" }) {
    // TODO: backend signup
    const newUser = { id: Date.now().toString(), email, role, name };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newUser));
    setUser(newUser);
    return newUser;
  }

  function logout() {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
  }

  return <AuthContext.Provider value={{ user, login, signup, logout }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
