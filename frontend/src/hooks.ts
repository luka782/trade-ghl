import { useEffect, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'

export function useSessionState<T>(
  key: string,
  initialValue: T | (() => T),
): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = window.sessionStorage.getItem(key)
      if (stored !== null) {
        return JSON.parse(stored) as T
      }
    } catch {
      // Storage can be unavailable in privacy-restricted environments.
    }

    return typeof initialValue === 'function'
      ? (initialValue as () => T)()
      : initialValue
  })

  useEffect(() => {
    try {
      window.sessionStorage.setItem(key, JSON.stringify(value))
    } catch {
      // The dashboard remains usable even if storage is unavailable.
    }
  }, [key, value])

  return [value, setValue]
}
