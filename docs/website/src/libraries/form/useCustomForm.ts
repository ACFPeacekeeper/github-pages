import { useForm } from '@tanstack/react-form';

/**
 * Shared TanStack Form helper for docs-site demos (parity with main site).
 */
export function useCustomForm<TValues extends Record<string, unknown>>(options: {
  defaultValues: TValues;
  onSubmit?: (props: { value: TValues }) => void | Promise<void>;
}) {
  return useForm({
    defaultValues: options.defaultValues,
    onSubmit: options.onSubmit
      ? async ({ value }) => {
          await options.onSubmit?.({ value: value as TValues });
        }
      : undefined,
  });
}

export { useForm } from '@tanstack/react-form';
