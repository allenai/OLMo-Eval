import { useEffect, useMemo } from "react";
import { assignSlots, rememberSlots, useColorMap } from "./prefs";

/** Categorical slots for subjects shown together on one page (see assignSlots). */
export function useSubjectSlots(keys: string[]): Record<string, number | null> {
  const prefs = useColorMap();
  const signature = keys.join(",");
  const { slots, updates } = useMemo(
    () => assignSlots(signature ? signature.split(",") : [], prefs),
    [signature, prefs],
  );
  useEffect(() => rememberSlots(updates), [updates]);
  return slots;
}
