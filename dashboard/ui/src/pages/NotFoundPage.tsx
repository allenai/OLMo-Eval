import { Compass } from "lucide-react";
import { AppLink } from "@/components/AppLink";
import { Button, EmptyState } from "@/components/primitives";
import { openPalette } from "@/components/shell/registry";

export function NotFoundPage({ what }: { what?: string }) {
  return (
    <div className="page">
      <EmptyState
        icon={<Compass />}
        title={what ? `${what} not found` : "Page not found"}
        actions={
          <>
            <Button onClick={() => openPalette()}>Search</Button>
            <AppLink href="/">
              <Button variant="ghost">Go home</Button>
            </AppLink>
          </>
        }
      >
        {what ? "It may have been deleted, or the link is wrong." : "The link may be wrong or out of date."}
      </EmptyState>
    </div>
  );
}
